# MNIST-projects
A collection of projects involving the MNIST dataset.

##Getting Started
Run any of the programs by running the following on the command line:
```
python <program file> <reuse>
```
Set reuse to 'True' to reuse a saved model. Any other argument is treated as 'False', and any saved models will be overridden. 

Each program has several parameters that can be modified near the top of each file.

The helper folder contains a number of potentially useful models as well some additional utility functions.

###train_input.py

This program first trains a convolutional neural network to identify labels of MNIST digit images. Then, keeping the model parameters fixed, it trains the input to maximize the probability of each of the ten labeled digits. These trained input images are then fed back into the model training step and are labeled as a eleventh 'trained image' label. This process is repeated so that the trained input images more and more closely resemble each of the ten digits.

###unbiased_digits.py

The goal of this program is to produce an encoding of each MNIST digit image that contains information on whether or not a digit contains a loop (0, 4, 6, 8, 9) or not (1, 2, 3, 5, 7) while not containing information on the parity of the digit. Note that looped digits are more even than odd, so the encoded representation must balance these two goals. The encoded representation is the output of a convolutional neural network. Loop and parity information is then derived from two seperate fully connected layers. 

Training happens in two parts. One optimizer trains the fully connected layer determining a digit's parity. Another optimizer trains the remainder of the network by attempting to determine whether a digit is a loop digit while simultaneously increasing the loss of the parity optimizer. The result is that the encoding contains information on a digit's loopiness while minimizing the information on a digit's parity.

###autoencode.py

This program creates an autoencoder for the MNIST digit images. Unlike a regular autoencoder, the encoder and decoder components are trained seperately with different objectives. The decoder is trained to reduce the difference between the output and input images. The encoding component is trained in order to reduce the difference between the encoded input image representation and the encoded representation of the output image when it is reencoded. The result is that the output images retain information that allows them to be approximately reencoded.
