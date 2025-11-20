# COMP-560-Final

The code and data in this repository can:
1) Train the Neural Network (hand-written digit classifer) with Training.py
2) Show an example of a random input and the final output Tensor of the NN with Usable_Model.py
3) Visualize the parameters in the first layer of the NN with Plot_parameters.py

# Data
The data used to train this model is the MNIST dataset, a large database of handwritten digits: 60000 for training and 10000 for testing. Each image is 28x28 pixels. 

# Training
To train the NN, run Training.py, saving the model to whichever path you specify (one model is already saved in the models folder). The program outputs each training epoch and the updated loss each time to the terminal.

# Random input
To test the model with some random input, run the Usable_Model.py, which will generate a random input from the data and show you the output tensor, and the neuron with highest activation. Make sure the model you choose to load is the right one. The program displays the random chosen image, and then shows the final output tensor of the model.

# Visualize weights
To visualize the weights of the first layer for each digit in the NN, run Plot_parameters.py, making sure you've loaded the preferred model. The program will output the importance map for each of the numbers for the first layer of the NN.

# Libraries

All libraries needed are listed in the requirements.txt.

# [Slide deck](https://docs.google.com/presentation/d/1Q0rcDVQ5jmizZW7BL7V_efnVCYnswZUQJV2-FygUVxw/edit?usp=sharing)
