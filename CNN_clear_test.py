from random import random, randint
import math
from time import sleep

def error(output, expected):
    if len(output) != len(expected):
        raise "length of the arrays output and expected are not the same"
    result = 0
    for x in range(len(output)):
        result += (expected[x]-output[x])**2
    return 1/2 * result


def random_array(x):
    result = []
    for a in range(x):
        result.append(random()-0.5)
    return result

def sigmoid(x):
    return 1 / (1+ math.exp(-x))

def sigmoid_I(x):
    return sigmoid(x)*(1.0-sigmoid(x))


class Neuron:
    def __init__(self, J_Length_int):
        self.weights = random_array(J_Length_int)


    def output_float(self, input_array):
        result = 0
        for n in range(len(self.weights)):
            result += (self.weights[n]*input_array[n])
        return sigmoid(result)



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
            #für jede Layer 1 mal
            error_signal_old = error_signal_new
            error_signal_new = []
            for N in range(len(reversed_layer[L].Neurons)):
                #für jedes Neuron einmal (mit n vielen gewichten)
                for W in range(len(reversed_layer[L].Neurons[N].weights)):
                    #Für jedes Gewicht im Neuron 1 mal (also für jedes Neuron in der schicht davor einmal)
                    
                    error_1 = sigmoid_I(reversed_layer[L].output[N])
                    #teil 1 ist bei beiden Formeln (hidden ud output neuron) gleich

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


NN = CNN([5,10, 10, 10, 5], 5)
input_data=[0.878, 0.4657, 0.098, 0.876, 0.123]
expected_array = [0.1, 0.1, 0.9, 0.9, 0.5]
output = NN.feed_forwoard(input_data)
print(output)
print(expected_array)

for x in range(10000):
    NN.train(input_data, expected_array)
    output_array = NN.feed_forwoard(input_data)

print(output_array)
print(expected_array)
print("\n---------------------------------------\n")

