# Facial Recognition

The objective is to use PyTorch and its neural network modelisation to tell the names of the people in a picture.

What I am really looking for, is to test different models and observe the challenges associated with using neural networks in the specific case of recognizing a face from its features.

Observations could be on: 
- accuracy of a model on a trained set, on an untrained set 
- influence of the number of hidden layers, number of neurons
- type of activation functions 
- overfitting issues
- type of neural networks
- learning rate
- ...

## Creating training sets
I use OpenCV and a pre-trained model to first detect the face elements of a picture. The training set generated is a json file containing a list of picture paths, containing face frames at specific positions and names.
The pre-trained model is imperfect, so sometimes frames rendered are not faces. This will have to be reflected on the training set. 

## Inputs and Outputs of a model
The inputs have to be consistent, they will be images of the same format representing the faces. So before each input, there will be a transformation of a selected frame (from the json training file) to a common image type. They are generated from the standardization process (cropping, resizing, pixels normalization).
The outputs will reflect the names of the persons to be recognized, or if it is an unknown person, or sometimes if it is not even a face.  

## Model training and optimization
This is where the model changes and optimization will be tested. I use PyTorch but due to hardware limitations, I am unable to use its CUDA functionalities. So I might have lengthy optimizations depending on what type of model I choose.
There are a varieties of possible models to be tested: multilayer perceptron, convoluted neural network, recurrent neural networks... On top of this I will test different optimization parameters.

## Recognizing faces in a picture
Based on a picture selected, a forward pass process using a selected model will define each frame and a name associated. The picture will then be displayed accordingly.




