import pyrealsense2 as rs
import numpy as np

def add_args(parser):
    parser.add_argument('--rs_bag', nargs='?', required=False, type=str, help='Path to optional input realsense .bag file to use instead of live camera stream')

def start_stream(args):

    pipeline = rs.pipeline()
    config = rs.config()

    rs_bag = args.rs_bag
    
    if rs_bag:
        config.enable_device_from_file(rs_bag, repeat_playback=True)
        config.enable_stream(rs.stream.depth, rs.format.z16)
        config.enable_stream(rs.stream.color, rs.format.rgb8)

    else:
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        # in the place where the original command was run, this file must be available
        device_config_json = open('hand_config.json', 'r').read()
        rs.rs400_advanced_mode(device).load_json(device_config_json)
        device.first_depth_sensor().set_option(rs.option.depth_units, 0.0001)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)

    profile = pipeline.start(config)
    if rs_bag:
        profile.get_device().as_playback().set_real_time(False)
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    depth_intrin = depth_profile.get_intrinsics()

    DIM_X = depth_intrin.width
    DIM_Y = depth_intrin.height

    FOCAL = depth_intrin.fx
    PP = np.array([depth_intrin.ppx, depth_intrin.ppy], dtype=np.float32)

    return pipeline, depth_intrin, DIM_X, DIM_Y, FOCAL, PP
