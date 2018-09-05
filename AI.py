from random import random
from math import exp
import json

#this function is used to create arrays with random floats between -0.5 and 0.5
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

#the method output float calculates the sum of all activations multiplied with there weights
#it returns a float
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
    #the method calculate output calculates the output of this whole layer. 
    #it takes the output of the Layer before (or the user given Input incase of an input Layer)
    #and returns an array with one item for each Neuron output this array can be given into the next Layer as Input
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

#In Fututure more Functionality will be coming
class NN:
    def __init__(self, L_Length_array=[10, 10, 5, 2], Input_Length_int=10, training_rate=0.05, reload=False):
        if not reload:
            self.Layers = []
            self.Input_Length_int = Input_Length_int
            self.training_rate = training_rate
            for n in range(len(L_Length_array)):
                if n != 0:
                    x = Layer(L_Length_array[n], L_Length_array[n-1])
                else:
                    x = Layer(L_Length_array[n], 1)
                self.Layers.append(x)


    #this function feeds the Input data into the CNN and returns its output
    #this function doesnt manipulate the Network
    def feed_forwoard(self, input_array):
        if len(input_array)!= self.Input_Length_int:
            raise "Inputdata doesnt match the given Length"
        else:
            for Layer in self.Layers:
                input_array = Layer.calculate_output(input_array)
            return input_array


    #this function includes the whole Traing for one Training step
    #you should use this in a "for" or "while" structure
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
                   
                
                
                
                    #|---------------------|
                    #|backpropagation stuff|
                    #|---------------------|
                    #better read abaout it at this place:
                    #https://de.wikipedia.org/wiki/Backpropagation#Fehlerminimierung
                    #its where I got the equations from
                    #---------------------------------------------------------------------------------
                    error_1 = sigmoid_I(reversed_layer[L].output[N])
     

                    if L == 0:
                        error_2 = reversed_layer[L].output[N]-expected_array[N]
                    else:
                        error_2 = 0
                        if len(reversed_layer) == L+1:
                            continue
                            #here we should take care about the input neurons and how they weight there inputs
                            #should they weight there inputs?
                        else:
                            for k in range(len(reversed_layer[L+1].Neurons)):
                                error_2 += (reversed_layer[L].Neurons[N].weights[k]*error_signal_old[k])
                    
                    error = -error_1*error_2
                    error_signal_new.append(error)
                    delta_w = self.training_rate*error*reversed_layer[L].output[N]
                    reversed_layer[L].Neurons[N].weights[W] += delta_w
                    #---------------------------------------------------------------------------------

    def save_NN(self, NN_file):
        f = open(NN_file, "wb")
        the_three_important = [self.Layers, self.training_rate, self.Input_Length_int]

        NN_as_string = json.dumps(the_three_important)
        f.write(NN_as_string)
        f.close()

    def reload_NN(self, NN_file):
        f = open(NN_file, "rb")
        NN_as_string = f.read()
        data = json.loads(NN_as_string)
        self.Input_Length_int = data[2]
        self.training_rate = data[1]
        self.Layers = data[0]


