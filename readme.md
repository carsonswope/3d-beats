Using random decision forests to identify hand pixels!

# Random forests!
This is an implementation of depth image pixel classification via random forests, as outlined in *Real-Time Human Pose Recognition in Parts from Single Depth Images* (look in docs). This is the technique used in the original microsoft kinect for realtime human pose estimation.

Labeled training data is obtained by using paint to color various sections of the object (hand) and recording both depth and color video streams. Additionally, forests can be stacked on top of one another, for segmenting the classification task into sub-tasks.

![Hand Classifier](rdf.gif)

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
  - CUDA binaries
  - model files
