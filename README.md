Covid X-Ray Image Classification

# Dataset is currently not provided
The dataset folder includes ground-truth images of each class (310 negative and 150 positive). --> 
The task is to provide an AI solution that can classify images as a positive or negative covid case.

# About the Model
Transfer Learning - Scenario with ConvNet as fixed feature extractor

Here, we will freeze the weights for all of the network except that of the final fully connected layer. This last fully connected layer is replaced with a new one with random weights and only this layer is trained.

# Requirements
Installation of Pytorch - Python 3.8.
