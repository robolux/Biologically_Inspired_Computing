# Main for Assignment 1 in Biologically Inspired Computing
# Hunter Phillips

from movements import data_call
import mlp
import numpy as nmp
import math

filename = 'data/movements_day1-3.dat'
train, train_targets, valid, valid_targets, test, test_targets = data_call(filename=filename)

print('\nWelcome to the Multilayer Perceptron (MLP) Robotic Prosthetic Hand Interface')
print('******************************************************************************')
print('The program will classify electromyographic (EMG) signals')
print('corresponding to various hand motions using the MLP.')
print('The network will contain one hidden layer with a')
print('user selected amount of nodes and iterations.\n')

hidn_i     = int(raw_input('\nHow many hidden nodes do you want to use for the single hidden layer?: '))
iterations = int(raw_input('\nHow many iterations do you want the MLP to run?: '))

print('\nThere is an option to show advanced options allowing customization that include:')
print('changing eta, beta, bias, and momentum values to experiment.')
print('This is not reccomended for beginners (proceed with caution)\n')

bx = False
while bx == False:
    adv = raw_input('Would you like to activate advanced mode? (Yes or No): ')
    adv.replace(" ", "") # cleanup messy user input

    if (str.upper(adv) == "YES"):
        eta_i      = float(raw_input('What is your desired eta?: '))
        beta_i     = float(raw_input('What is your desired beta?: '))
        bias_i     = float(raw_input('What is your desired bias?: '))
        momentum_i = float(raw_input('What is your desired momentum?: '))
        bx = True

    elif (str.upper(adv) == "NO"):
        eta_i       = 0.1
        beta_i      = 1.0
        bias_i      = -1.0
        momentum_i  = 0.0
        bx = True
    else:
        print('User Input not recognized as option, please try again.')

print('\nRunning...')
for i in range(iterations):
	active = mlp.mlp(train, train_targets, hidn_i, beta = beta_i, eta = eta_i, bias = bias_i, momentum = momentum_i)
	active.earlystopping(train, train_targets, valid, valid_targets)
c_matrix = active.confusion(test, test_targets)


print('\nResulting Confusion Matrix')
print("************************************")
print(c_matrix)

# Output Class Prediction Percentages
print('\nClass Prediction Percentages')
print("************************************")
average = 0
for m, n in enumerate(c_matrix.transpose()):
	value = (n[m]/nmp.sum(n))*100
	if math.isnan(value):
		value = 0

	print("{} : {}%".format(m+1,value))
	average+=value

# Output Average Percentage
print('The average percentage is {}%\n'.format(average/8))
