'''
The Convolutional Neural Network (CNN) we are implementing here with PyTorch is the seminal LeNet architecture,
first proposed by one of the grandfathers of deep learning, Yann LeCunn.

By today’s standards, LeNet is a very shallow neural network, consisting of the following layers:
(CONV => RELU => POOL) * 2 => FC => RELU => FC => SOFTMAX
'''

# import the necessary packages
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax # Used when building our softmax classifier to return the predicted probabilities of each class
from torch import flatten

# Implement our LeNet class using PyTorch
class LeNet(Module):
	def __init__(self, numChannels, classes):
		# call the parent constructor
		super(LeNet, self).__init__() # calls the parent constructor (i.e., Module) which performs a number of PyTorch-specific operations
		# initialize first set of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
			kernel_size=(5, 5))
		self.relu1 = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize second set of CONV => RELU => POOL layers
		self.conv2 = Conv2d(in_channels=20, out_channels=50,
			kernel_size=(5, 5))
		self.relu2 = ReLU()
		self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize first (and only) set of FC => RELU layers
		self.fc1 = Linear(in_features=800, out_features=500)
		self.relu3 = ReLU()
		# initialize our softmax classifier
		self.fc2 = Linear(in_features=500, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)
	'''
	By building our model as a class we can easily:
	    - Reuse variables
	    - Implement custom functions to generate subnetworks/components (used very often when implementing more complex networks, such as ResNet, Inception, etc.)
	    - Define our own forward pass function
	'''

	'''
	At this point all we have done is initialized variables. These variables are essentially placeholders. PyTorch has absolutely no idea 
	what the network architecture is, just that some variables exist inside the LeNet class definition.

	To build the network architecture itself (i.e., what layer is input to some other layer), we need to override the forward
	method of the Module class.
	'''
	def forward(self, x):
		# x = the batch of input data to the network
		# pass the input through our first set of CONV => RELU =>
		# POOL layers
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
		# pass the output from the previous layer through the second
		# set of CONV => RELU => POOL layers
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)
		# flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
		x = flatten(x, 1)
		x = self.fc1(x)
		x = self.relu3(x)
		# pass the output to our softmax classifier to get our output
		# predictions
		x = self.fc2(x)
		output = self.logSoftmax(x)
		# return the output predictions
		return output
