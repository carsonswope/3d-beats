import numpy as np
from PIL import Image
import pycuda.gpuarray as cu_array
import pycuda.driver as cu
import nvcomp

from util import sizeof_fmt, PagelockedCounter

class CompressedBlocksDynamic():
    def __init__(self, num_blocks, num_images_per_block, img_dims, img_dtype, name):
        self.num_blocks = num_blocks
        self.block_shape = (num_images_per_block, img_dims[1], img_dims[0])
        self.img_dtype = img_dtype
        self.block_size = num_images_per_block * img_dims[0] * img_dims[1] * np.dtype(img_dtype).itemsize

        self.compressor = nvcomp.CascadedCompressor(nvcomp.nvcompType_t.NVCOMP_TYPE_INT, 2, 1, True)
        self.temp_size = PagelockedCounter()
        self.compressor_output_max_size = PagelockedCounter()
        self.compressor.configure(self.block_size, self.temp_size.ptr, self.compressor_output_max_size.ptr)

        # temp shared between compressor and decompressor
        self.temp_cu = cu_array.GPUArray((self.temp_size(),), dtype=np.uint8)
        self.compressor_output_cu = cu_array.GPUArray((self.compressor_output_max_size(),), dtype=np.uint8)
        self.compressor_output_size = PagelockedCounter()

        self.expected_compressor_output_max_size = 20000000 # 20 mb?
        self.compressed_blocks = cu_array.GPUArray((num_blocks, self.expected_compressor_output_max_size,), dtype=np.uint8)
        self.compressed_block_sizes = [0 for _ in range(num_blocks)]

        self.decompressor = nvcomp.CascadedDecompressor()
        self.decompressor_temp_size = PagelockedCounter()
        self.decompressor_output_size = PagelockedCounter()

        print('Initialized dynamic image blocks: ', name)
        print('  ' + str(self.num_blocks), 'blocks of', str(self.block_shape))
        print('  uncompressed size:      ', sizeof_fmt(self.num_blocks * self.block_size))
        print('  est. compressed size:   ', sizeof_fmt(self.expected_compressor_output_max_size * self.num_blocks))
        print('  temp size:              ', sizeof_fmt(self.temp_size()))


    def write_block(self, block_number, block_cu):

        assert self.block_size == block_cu.size * block_cu.itemsize
        assert block_number >= 0 and block_number < self.num_blocks
        assert self.block_shape == block_cu.shape

        self.compressor.compress_async(
            block_cu.ptr,
            self.block_size,
            self.temp_cu.ptr,
            self.temp_cu.size,
            self.compressor_output_cu.ptr,
            self.compressor_output_size.ptr)

        cu.Context.synchronize()
        
        compressed_size = self.compressor_output_size()
        assert compressed_size <= self.expected_compressor_output_max_size

        self.compressed_blocks[block_number, 0:compressed_size].set(self.compressor_output_cu[0:compressed_size])
        self.compressed_block_sizes[block_number] = compressed_size


    def get_block(self, block_number, block_cu):

        assert self.block_size == block_cu.size * block_cu.itemsize
        assert block_number >= 0 and block_number < self.num_blocks
        assert self.block_shape == block_cu.shape

        self.decompressor.configure(
            self.compressed_blocks[block_number].ptr,
            self.compressed_block_sizes[block_number],
            self.decompressor_temp_size.ptr,
            self.decompressor_output_size.ptr)

        cu.Context.synchronize()

        assert self.decompressor_output_size() == self.block_size

        if self.decompressor_temp_size() > self.temp_size():
            self.temp_size.set(self.decompressor_output_size())
            del self.temp_cu
            self.temp_cu = cu_array.GPUArray((self.temp_size,), dtype=np.uint8)
            print('Reallocated temp space for decompressor. New size: ', sizeof_fmt(self.temp_size))
        
        self.decompressor.decompress_async(
            self.compressed_blocks[block_number].ptr,
            self.compressed_block_sizes[block_number],
            self.temp_cu.ptr,
            self.decompressor_temp_size(),
            block_cu.ptr,
            self.block_size)


class CompressedBlocksStatic():
    def __init__(self, num_blocks, num_images_per_block, img_dims, get_img_path, images_name):

        self.num_blocks = num_blocks
        self.block_shape = (num_images_per_block, img_dims[1], img_dims[0])

        block_np = np.zeros(self.block_shape, np.uint16)
        uncompressed_block_cu = cu_array.to_gpu(block_np)
        block_size = block_np.size * block_np.itemsize

        compressor = nvcomp.CascadedCompressor(nvcomp.nvcompType_t.NVCOMP_TYPE_USHORT, 2, 1, True)
        compressor_temp_size = PagelockedCounter()
        compressor_output_max_size = PagelockedCounter()
        compressor.configure(block_size, compressor_temp_size.ptr, compressor_output_max_size.ptr)

        compressor_temp_cu = cu_array.GPUArray((compressor_temp_size(),), dtype=np.uint8)
        compressor_output_cu = cu_array.GPUArray((compressor_output_max_size(),), dtype=np.uint8)
        compressor_output_size = PagelockedCounter()

        # first store compressed blocks on CPU, then put together and transfer to GPU
        compressed_blocks = []

        for i in range(num_blocks):
            for j in range(num_images_per_block):
                img_idx = (i * num_images_per_block) + j
                block_np[j] = np.array(Image.open(get_img_path(img_idx))).astype(np.uint16)

            # first compress depth
            uncompressed_block_cu.set(block_np)
            compressor.compress_async(
                uncompressed_block_cu.ptr,
                block_size,
                compressor_temp_cu.ptr,
                compressor_temp_size(),
                compressor_output_cu.ptr,
                compressor_output_size.ptr)

            cu.Context.synchronize()
            compressed_block_size = compressor_output_size()
            compressed_blocks.append(compressor_output_cu[0:compressed_block_size].get())

        # done using compressor..
        del compressor_output_size
        del compressor_output_cu
        del compressor_temp_cu
        del compressor

        all_compressed_blocks_size = sum([b.shape[0] for b in compressed_blocks])

        self.block_idxes = [] # (start_idx, length)

        current_idx = 0

        all_compressed_blocks_cpu = cu.pagelocked_zeros((all_compressed_blocks_size,), dtype=np.uint8)

        for i in range(num_blocks):
            compressed_block_size = compressed_blocks[i].shape[0]
            self.block_idxes.append((current_idx, compressed_block_size))
            all_compressed_blocks_cpu[current_idx : current_idx+compressed_block_size] = compressed_blocks[i]
            current_idx += compressed_block_size

        self.all_compressed_blocks_cu = cu_array.to_gpu(all_compressed_blocks_cpu)
        del all_compressed_blocks_cpu

        self.decompressor = nvcomp.CascadedDecompressor()
        self.decompressor_temp_size = PagelockedCounter()
        self.decompressor_output_size = PagelockedCounter()

        max_decompressor_temp_size = 0

        for i in range(num_blocks):

            # first depth
            block_idx, block_compressed_size = self.block_idxes[i]
            self.decompressor.configure(
                self.all_compressed_blocks_cu[block_idx].ptr,
                block_compressed_size,
                self.decompressor_temp_size.ptr,
                self.decompressor_output_size.ptr)
            cu.Context.synchronize()
            assert self.decompressor_output_size() == block_size
            max_decompressor_temp_size = max(max_decompressor_temp_size, self.decompressor_temp_size())

        self.decompressor_temp_cu = cu_array.GPUArray((max_decompressor_temp_size,), dtype=np.uint8)

        print('Loaded images compressed: ', images_name)
        print('  ' + str(self.num_blocks), 'blocks of', str(self.block_shape))
        print('  uncompressed size:      ', sizeof_fmt(self.num_blocks * block_size))
        print('  compressed size:        ', sizeof_fmt(self.all_compressed_blocks_cu.size * self.all_compressed_blocks_cu.itemsize)) # itemsizes should be 1..
        print('  decompressor temp size: ', sizeof_fmt(self.decompressor_temp_cu.size * self.decompressor_temp_cu.itemsize))


    def get_block_cu(self, block_num, arr_out):

        assert block_num < self.num_blocks
        assert arr_out.shape == self.block_shape

        block_start_idx, block_size = self.block_idxes[block_num]

        # allocations already made, still have to configure though..
        self.decompressor.configure(
            self.all_compressed_blocks_cu[block_start_idx].ptr,
            block_size,
            self.decompressor_temp_size.ptr,
            self.decompressor_output_size.ptr)

        self.decompressor.decompress_async(
            self.all_compressed_blocks_cu[block_start_idx].ptr,
            block_size,
            self.decompressor_temp_cu.ptr,
            self.decompressor_temp_cu.size, # itemsize==1, uint8
            arr_out.ptr,
            arr_out.size * arr_out.itemsize)
