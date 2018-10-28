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

		def node_mult(var1, var2):
			mult = []
			hid_out_u = []
			hid_out = []
			c_size = nmp.shape(var2)
			for mm in range(len(var1)):
				hid_out_u = []
				for rx in range(c_size[1]):
					mult = []
					for tt in range(c_size[0]):
						yy = var1[mm][tt]
						ui = var2[tt][rx]
						mult_trash = yy * ui
						mult.append(mult_trash)
					hid_out_u.append(sum(mult))
				hid_out.append(hid_out_u)
			return hid_out

		for i in range(iterations):
			hiddenL, outputL = self.forward(inputs) # forward

			#Backpropagation
			outputdelta = (outputL - targets)*(outputL*(1-outputL))						 # sigmoid derivative function

			dim_sum = nmp.shape(self.output_l_weights)
			rearrange_ol = []
			for i2 in range(dim_sum[1]):
			    l2 = []
			    for row in self.output_l_weights:
			        l2.append(row[i2])
			    rearrange_ol.append(l2)
			rearrange_ol = nmp.asarray(rearrange_ol)


			hid_out = node_mult(outputdelta, rearrange_ol)

			hid_out = nmp.asarray(hid_out)

			hiddendelta = hid_out*(hiddenL*(1-hiddenL))

			# hiddendelta = outputdelta.dot(self.output_l_weights.T)*(hiddenL*(1-hiddenL)) # sigmod derivative function

			dim_olw = nmp.shape(hiddenL)
			rearrange_olw = []
			for i2 in range(dim_olw[1]):
			    l_olw = []
			    for row in hiddenL:
			        l_olw.append(row[i2])
			    rearrange_olw.append(l_olw)
			rearrange_olw = nmp.asarray(rearrange_olw)

			rearrange_olw = rearrange_olw*self.eta
			mult_olw = node_mult(rearrange_olw, outputdelta)
			mult_olw = nmp.asarray(mult_olw)
			self.output_l_weights -= mult_olw



			dim_hlw = nmp.shape(inputs)
			rearrange_hlw = []
			for i2 in range(dim_hlw[1]):
			    l_hlw = []
			    for row in inputs:
			        l_hlw.append(row[i2])
			    rearrange_hlw.append(l_hlw)
			rearrange_hlw = nmp.asarray(rearrange_hlw)

			rearrange_hlw = rearrange_hlw*self.eta
			mult_hlw = node_mult(rearrange_hlw, hiddendelta)
			mult_hlw = nmp.asarray(mult_hlw)
			self.hidden_l_weights -= mult_hlw

			# update weights
			# self.output_l_weights -= self.eta*hiddenL.T.dot(outputdelta)
			# self.hidden_l_weights -= self.eta*inputs.T.dot(hiddendelta)

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
