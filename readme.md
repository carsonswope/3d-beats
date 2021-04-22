Using random decision forests to identify hand pixels!

# Data generation:
Inside the `datagen` folder, there is a `.blend` (blender) scene set up. It contains a python script that can be run to generate a bunch of fake data.
The original hand model is downloaded from: https://www.youtube.com/watch?v=PW5dJVQe83U

# Training:


make split(pixels):
    Randomly generate N features (U, V, decision threshold)
    Pick feature which provides best split (via entropy)
    If 