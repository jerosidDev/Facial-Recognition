# Facial Recognition

The objective is to use PyTorch and its neural network modelisation to tell the names of the people in a picture.

What I am really looking for, is to try different models and observe the challenges associated with using neural networks in the specific case of a recognizing a face from its features.

Observations could be on: 
- accuracy of a model on a trained set, on an untrained set 
- influence of the number of hidden layers, number of neurons
- type of activation functions 
- overfitting issues
- type of neural networks
- learning rate

## Creating training sets
I will be using OpenCV to first detect the face elements of a picture. The training set will simply be a json file containing a list of picture paths, containing face frames at specific positions and names. 


