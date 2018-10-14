# Implementation of MLP using skeleton provided by instructors.
# All code in functions was written by Hunter Phillips
# for Assignment 2 in Biologically Inspired Computing.

import numpy as nmp
import math

class mlp:
	nmp.seterr(over='ignore')
	nmp.seterr(invalid='ignore')

	def __init__(self, inputs, targets, nhidden, beta, eta, bias, momentum):
		self.beta = beta          # function parameter
		self.eta = eta            # learning rate
		self.bias= bias           # bias
		self.momentum = momentum  # momentum

		self.n_illa = inputs.shape[0]
		assert self.n_illa ==targets.shape[0]
		self.ninputs = inputs.shape[1]+1
		self.nhidden = nhidden+1
		self.outputs= targets.shape[1]
		#set random weights.
		self.hidden_l_weights= 2*nmp.random.random((self.ninputs, self.nhidden))-1
		self.output_l_weights = 2*nmp.random.random((self.nhidden, self.outputs))-1

	def earlystopping(self, inputs, targets, valid, validtargets):

		#bias on inputs
		inputb  = nmp.ones((inputs.shape[0], 1))*self.bias
		inputb2 = nmp.concatenate((inputb, inputs), axis=1)

		#bias on valid
		validb = nmp.ones((valid.shape[0], 1))*self.bias
		valid2 = nmp.concatenate((validb, valid), axis=1)

		err = float('inf')
		m_err=0.01
		it = 0

		while  err > m_err:
			self.train(inputb2, targets)
			hiddenL, outputR =  self.forward(valid2) # forward
			err_u = nmp.sum((outputR-validtargets)**2)/self.n_illa

			if err_u<err:
				err=err_u
			else:
				break
			it +=1

	def train(self, inputs, targets, iterations=100):

		for i in range(iterations):
			hiddenL, outputL = self.forward(inputs) # forward

			#Backpropagation
			outputdelta = (outputL - targets)*(outputL*(1-outputL))						 # sigmoid derivative function
			hiddendelta = outputdelta.dot(self.output_l_weights.T)*(hiddenL*(1-hiddenL)) # sigmod derivative function

			# update weights
			self.output_l_weights -= self.eta*hiddenL.T.dot(outputdelta)
			self.hidden_l_weights -= self.eta*inputs.T.dot(hiddendelta)

	def forward(self, inputs):
		#run forward

		hiddenL = 1/(1+nmp.exp(-self.beta*(nmp.dot(inputs, self.hidden_l_weights)))) #activation with sigmoid function
		hiddenL[:, 0] = self.bias
		outputL = 1/(1+nmp.exp(-self.beta*(nmp.dot(hiddenL, self.output_l_weights)))) #activation with sigmoid function

		return hiddenL, outputL

	def confusion(self, inputs, targets):
		#include bias on the inputs
		inputb = nmp.ones((inputs.shape[0], 1))*self.bias
		inputb2 = nmp.concatenate((inputb, inputs), axis=1)

		hiddenL, predicted = self.forward(inputb2)

		if self.outputs ==1:
			predicted = nmp.array(predicted >= 0.5, dtype = int )
		else:
			predicted = nmp.array(predicted == nmp.array([nmp.max(predicted, axis = 1)]).T)

		c_matrix = nmp.zeros((self.outputs, self.outputs))

        # form confusion matrix to be returned
		for targetrow, outputrow in zip(targets,predicted):
			targetindex = nmp.where(targetrow == 1)[0][0]
			outputindex = nmp.where(outputrow == 1)[0][0]
			c_matrix[targetindex, outputindex] += 1

		return c_matrix
