Using random decision forests to identify hand pixels!

# Random forests!
This is an implementation of depth image pixel classification via random forests, as outlined in *Real-Time Human Pose Recognition in Parts from Single Depth Images* (look in docs). This is the technique uesd by the Microsoft Kinect team to train their human pose estimation model.

One major difference is that they trained their model on synthetic data, but I'm training the model using real recordings of my hand, with skin paint to label different sections.

This is still a work in process, but here's a gif of the classifier in action:

![Hand Classifier](rdf.gif)

# TODOs:

- velocity:
  - ignore taps below a certain velocity
  - option to vary midi notes' velocity with velocity of tap
- ignore fingertips that are not featured prominently / or have high variance from mean shift
- auto-tuning of plane / z threshold. should be able to determine best z threshold automatically
- add thumb sections to multi-level RDF architecture.
- OR simpler 2-stage RDF architecture that simply identifies:
  - 1. fingertip OR {rest of hand}
  - 2. identify which fingertip
