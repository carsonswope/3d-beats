# 3d Beats v2!

![Hand Classifier](rdf.gif)

At the core, this is an implementation of depth image pixel classification via randomized decision forests, as outlined in *Real-Time Human Pose Recognition in Parts from Single Depth Images* (look in docs). This is the technique used in the original microsoft kinect for realtime human pose estimation.

Labeled training data is obtained by using skin paint to color various sections of the subject (hand) and recording both depth and color video streams. Additionally, forests can be stacked on top of one another, for segmenting the classification task into sub-tasks.

To generate fingertip positions, the mean shift algorithm is used to find the center of each classification region. The original depth image is then sampled at the determined pixel, to find the height of each fingertip above the plane. 

## Instructions for installation / development.

Unfortunately, it's not very well packaged for easy distribution at the moment. To start, you will need:

- [microsoft visual studio 2019](https://visualstudio.microsoft.com/downloads/) (2019). Unfortunately you have to download the whole thing just to get the compiler toolkit
- [cuda toolkit](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Windows&target_arch=x86_64) (11.3 ? Maybe other versions would work)
- [conda python environment manager](https://docs.conda.io/en/latest/miniconda.html)

When visual studio is installed it doesn't automatically put the compiler and other build tools on the PATH so you have to run a script which modifies the correct environment variables, etc. Visual studio comes with `.bat` files for this which only run in the `cmd` command prompt, but I prefer powershell, so I use this function from powershell to re-enter the powershell terminal.

```ps1
function vs2019-64 { cmd /k "`"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat`" & pwsh" }
```

Now run the newly declared `vs2019-64` to enter the MSVC command prompt. You should now be able to run `cl` in the terminal to access the C/C++ compiler. You should also be able to run `nvcc --version` to access 
the CUDA compiler. Basically this is so CUDA kernels can be compiled at runtime. Eventually it will be possible to distribute precompiled binaries, but haven't figured exactly how to do that yet in python.

Now activate conda. This is another powershell function which I use to enter the conda environment:

```ps1
function conda-terminal {
	&'C:\Users\Carson\miniconda3\shell\condabin\conda-hook.ps1'
	conda activate 'C:\Users\Carson\miniconda3'
}
```

So now run `conda-terminal` (from still inside the same terminal window). Create a new conda environment, and install dependencies. It is important to be on python 3.7 because I compiled some dependencies individually, and the compiled packages were configured for 3.7 specifically.

```
conda create -n 'env_3d_beats' python=3.7.10
conda activate env_3d_beats
pip install -r requirements.txt
```

Now figure out where your `site-packages` directory is:

`python -c 'import site; print(site.getsitepackages())'`

And copy everything in the `deps` directory to that directory:

`cp .\deps\* C:\Users\Carson\miniconda3\envs\env_3d_beats\Lib\site-packages\.`

Unfortunately, `pycuda` must be built from source to get opengl interop. This part can be run from anywhere, doesn't have to be in the repo directory. But make sure you still are running these commands from inside the correct conda python environment.

```
git clone git clone https://github.com/inducer/pycuda.git
cd pycuda
git submodule update --init
python ./configure.py --cuda-enable-gl
python setup.py build
python setup.py install
```

Some cuda kernels rely on GLM (library for OpenGL mathematics). The repo needs to be downloaded, but no further build is necessary.
```
cd src/cuda/deps
git clone https://github.com/g-truc/glm.git
```

Last but not least, I would recommend downloading the latest [RealSense viewer release](https://github.com/IntelRealSense/librealsense/releases) and use it to upgrade the firmware to the latest D415 fw. Or skip this step but, you know, in case of debugging. If you are having issues, you also may want to open up realsense viewer and confirm that the 848x480 depth stream at 90 FPS works there. Sometimes I find that the USB cord needs to be plugged in the 'other' way to the camera to get full USB 3 support (required for 90 FPS).

Now for the fun part. Mount the camera above the desk maybe 12-13" up. It should be across the desk from you, pointed down at the desk but not directly down. Left-to-right should be the x direction of the camera, and the USB cord should be sticking out to your left, camera's right. This is just to make sure your hands are in the expected orientation, as the model is very sensitive to that kind of thing. 

First, run the simple demo of the RDF in action, just to make sure the basic pipeline is working. It is expecting a right hand in the view, and nothing else. Make sure the color rendering of the RDF output appears to be accurately identifying different sections of the hand.

(Make sure you are in the root of this repo when running these commands. It will probably take ~30 seconds or so to start up.)

`python ./src/run_live_layered.py -cfg .\model\model_cfg.json`

Okay, so now the real thing! The idea is that each fingertip and thumb is assigned it's own MIDI note. Keep your left and right hands apart from eachother, and tap on the desk with flat fingers (hope to support more piano-style type fingering soon..). As you tap on the desk, the tap thresholds will calibrate to each finger - so expect performance to improve after tapping with each fingertip a few times. If you open up the 'L' and 'R' thingies in the UI, you can see the graph of the estimated heights of each fingertip over time, and a visual indication of where the thresholds are and whether the note is on or off. Fire up the DAW and have fun!

`python ./src/3d_bz.py -cfg .\model\model_cfg.json`

# TODOs:

- velocity:
  - option to vary midi notes' velocity with velocity of tap
- ignore fingertips that are not featured prominently / or have high variance from mean shift
- handle when mean shift has identified a pixel which is not part of the depth image (0 depth!)
- auto-tuning of plane / z threshold. should be able to determine best z threshold automatically
- make new model. simpler 2-stage RDF architecture:
  - 1. fingertip OR {rest of hand} OR {thumb tip?} (2-3 classes)
  - 2. identify which fingertip (4-5 classes)
- make midi selection more robust, / UI
- figure out how to build standalone executable/installer..
  - python w/ environment
  - custom package: cpp_grouping
  - cuda precompilation:
    - fatbins working for cu files written in this repo
    - pycuda uses autogeneration of other cuda code which has to be compiled by nvcc at runtime ( fill(), anything else? )
    - also cuRAND (as exposed by pycuda) also apparently requires runtime nvcc invocation
  - model files
 
