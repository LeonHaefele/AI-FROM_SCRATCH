from random import random
from math import exp

#this function is used to create arrays with random floats 
#x = the length of the array
def random_array(x):
    result = []
    for a in range(x):
        result.append(random()-0.5)
    return result
#the sigmoid function is the actiavtion function of the neuron
def sigmoid(x):
    return 1 / (1+ exp(-x))
# sigmoid_I is the derivation of sigmoid its nessecary for the backpropagation
def sigmoid_I(x):
    return sigmoid(x)*(1.0-sigmoid(x))


#------------------------------------------------------------------------------------------
#this class defines a neuron whichs most important task is, to hold an array with weights
#one weight for each neuron in the Layer before
class Neuron:
    def __init__(self, J_Length_int):
        self.weights = random_array(J_Length_int)


    def output_float(self, input_array):
        result = 0
        for n in range(len(self.weights)):
            result += (self.weights[n]*input_array[n])
        return sigmoid(result)


#------------------------------------------------------------------------------------------
#this class defines a Layer. Its an array of neurons
#with the Layer function you dont need the neuron function in daylie use
class Layer:
    def __init__(self, L_Length_int, J_Length_int):
        self.Neurons = []
        self.L_Length_int = L_Length_int
        self.output = []
        for a in range(L_Length_int):
            n = Neuron(J_Length_int)
            self.Neurons.append(n)

    def calculate_output(self, input_array):
        self.output = []
        for a in range(len(self.Neurons)):
            res = self.Neurons[a].output_float(input_array)
            self.output.append(res)

        return self.output



#------------------------------------------------------------------------------------------
#the CNN decribes an array of Layers
#you can use this class to 
#-create a CNN
#-train it
#read its output

#if i got the time there will be new functions coming
class CNN:
    def __init__(self, L_Length_array, Input_Length_int, training_rate=0.05):
        self.Layers = []
        self.Input_Length_int = Input_Length_int
        self.training_rate = training_rate
        for n in range(len(L_Length_array)):
            if n != 0:
                x = Layer(L_Length_array[n], L_Length_array[n-1])
            else:
                x = Layer(L_Length_array[n], 1)
            self.Layers.append(x)
    
    def feed_forwoard(self, input_array):
        if len(input_array)!= self.Input_Length_int:
            raise "Inputdata doesnt match the given Length"
        else:
            for Layer in self.Layers:
                input_array = Layer.calculate_output(input_array)
            return input_array

    def train(self, input_array, expected_array):
        self.feed_forwoard(input_array)

        reversed_layer = self.Layers[::-1]
        error_signal_new = []
        for L in range(len(reversed_layer)):
            #one time for every Layer in the Network
            error_signal_old = error_signal_new
            error_signal_new = []
            for N in range(len(reversed_layer[L].Neurons)):
                #one Time for every Neuron in the Layer
                for W in range(len(reversed_layer[L].Neurons[N].weights)):
                    #one Time for every Weight in the Neuron
                    
                    #backpropagation stuff
                    #better read abaout it at this place:
                    #https://de.wikipedia.org/wiki/Backpropagation#Fehlerminimierung
                    #its where I got the equations from
#---------------------------------------------------------------------------------
                    error_1 = sigmoid_I(reversed_layer[L].output[N])
                    

                    #teil 2 ist anders
                    if L == 0:
                        error_2 = reversed_layer[L].output[N]-expected_array[N]
                    else:
                        error_2 = 0
                        if len(reversed_layer) == L+1:
                            continue
                            #error_2 += (reversed_layer[L].Neurons[N].weights[0])
                        else:
                            for k in range(len(reversed_layer[L+1].Neurons)):
                                error_2 += (reversed_layer[L].Neurons[N].weights[k]*error_signal_old[k])
                    
                    error = -error_1*error_2
                    error_signal_new.append(error)
                    delta_w = self.training_rate*error*reversed_layer[L].output[N]
                    reversed_layer[L].Neurons[N].weights[W] += delta_w
#-------------------------------------------------------------------------------------------------------
