Using random decision forests to identify hand pixels!

# Random forests!
This is an implementation of depth image pixel classification via random forests, as outlined in *Real-Time Human Pose Recognition in Parts from Single Depth Images* (look in docs).

# Data generation:
Inside the `datagen` folder, there is a `.blend` (blender) scene set up. It contains a python script that can be run to generate a bunch of fake data.
The original hand model is downloaded from: https://www.youtube.com/watch?v=PW5dJVQe83U

# TODOs:

- ability to combine multiple datasets into a single dataset (all sets need same number of classes, and a mapping between classes between sets).
  this will be another script that is run after live_data_convert.py (need to manually map between classes of different starting datasets)
- random scaling/noise/etc. added to input data for more robustness
- smoothing of generated input labels data so as to improve quality of training data
- multi-tiered RDF architecture. Shouldn't require major tweaks to the decision forest, but need to update live_data_convert to handle the mask generation, and a new run_live for more advanced decision tree evaluation chains
    - 1st tree: 4 classes (fingers, thumb, hand, arm)
    - No further processing for hand & arm.
    - For fingers, use that as mask for 2 separate decision trees, 1st: distinguish fingers 1,2,3,4, 2nd: distinguish bones 1,2,3 across all fingers.
      Use combinations of fingers & bones to generate positions of all 12 finger bone positions
    - For thumb, use that as a mask for 1 separate decision tree, trained to distinguish bones 1,2,(3?) of the thumb