Using random decision forests to identify hand pixels!

# Random forests!
This is an implementation of depth image pixel classification via random forests, as outlined in *Real-Time Human Pose Recognition in Parts from Single Depth Images* (look in docs). This is the technique uesd by the Microsoft Kinect team to train their human pose estimation model.

One major difference is that they trained their model on synthetic data, but I'm training the model using real recordings of my hand, with skin paint to label different sections.

This is still a work in process, but here's a gif of the classifier in action:

![Hand Classifier](rdf.gif)

# TODOs:

- apply gaussian filter to incoming depth image for smoothing and downsampling (create /2, /4, /8 etc. mipmaps)
- copy smaller depthmap back to cpu, use 'paint fill' to distinguish different 'hands' (contiguous sections in depth image after filtering by plane)
- for each determined region:
  - apply a right vs left classifier (on small mipmap). this can be trained on original training data (which are all right hands), and flip every other frame.
  - if left, flip image before feeding to classifier
  - apply full 16 class classifier pipeline, mean shift to find centers.
  - make sure the main 'hand center' class features prominently (if not, discard region).
  - make sure to also ignore fingertips that are not featured prominently

- auto-tuning of plane / z threshold. should be able to determine best z threshold automatically
- add thumb sections to multi-level RDF architecture
