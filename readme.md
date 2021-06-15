Using random decision forests to identify hand pixels!

# Random forests!
This is an implementation of depth image pixel classification via random forests, as outlined in *Real-Time Human Pose Recognition in Parts from Single Depth Images* (look in docs).

# Data generation:
Inside the `datagen` folder, there is a `.blend` (blender) scene set up. It contains a python script that can be run to generate a bunch of fake data.
The original hand model is downloaded from: https://www.youtube.com/watch?v=PW5dJVQe83U

# TODOs:

- auto-tuning of plane / z threshold. should be able to determine best z threshold automatically
- random scaling/noise/etc. added to input data for more robustness
- smoothing of generated input labels data so as to improve quality of training data
- add thumb sections to multi-level RDF architecture