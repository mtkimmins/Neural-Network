#TODO refactor all this shit

#------------IMPORTS-----------------
import numpy as np
import math
import random


#-------------CLASSES-----------------
#####################
#   NEURAL NETWORK  #
#####################
class Network:
    def __init__(self, layers:list):
        self.numb_layers = len(layers)
        self.layers = layers

        self.activation_function = Sigmoid
        self.learning_rate = 0.1


        self.pre_activations = [] #these are row vectors
        self.activations = [] #these are row vectors
        self.weights = [] #multidimensional matrix
        self.biases = [] #these are row vectors

        for layer in layers:
            self.pre_activations.append(np.zeros(shape = (layer,1)))
            self.activations.append(np.zeros(shape = (layer,1)))

        rng = np.random.default_rng()
        for i in range(self.numb_layers-1):
            self.weights.append(rng.uniform(0,1, size=(self.layers[i+1], self.layers[i])))
            self.biases.append(rng.uniform(0,1, size=(self.layers[i+1],1)))
    
    def feedforward(self, data:np.ndarray):
        r_data = np.reshape(data, (784,1))
        #check that the array is the same size as the input layer
        #assign the array as the input layer
        #prop forward until you get an output
        #make branchoff matrices and store them in the appropriate place (PRE-ACTIVATIONS or ACTIVATIONS)
        if not self.can_insert_data(r_data, self.activations[0]): 
            raise Exception("Incompatible sizes: {0} = data shape, {1} = input layer shape".format(r_data.shape, self.activations[0].shape))
        
        layer:np.ndarray = r_data.copy()
        self.activations[0] = r_data.copy()
        self.pre_activations[0] = r_data.copy()
        propagations = self.numb_layers-1

        for i in range(propagations):
            #indices
            f_vec_i = i+1

            #multiply by weights
            layer = self.weights[i] @ layer
            #add biases
            layer += self.biases[i]

            #save pre-activation state
            if self.can_insert_data(layer, self.pre_activations[f_vec_i]):
                self.pre_activations[f_vec_i] = layer.copy()
            np.apply_over_axes(self.activation_function.function, layer, [0,layer.shape[0]])
            if self.can_insert_data(layer, self.activations[f_vec_i]):
                self.activations[f_vec_i] = layer.copy()

    def backpropagate(self, label:np.ndarray) -> tuple:
        r_label = np.reshape(label, (10,1))
        if not r_label.shape == self.activations[-1].shape: 
            raise Exception("Incompatible sizes: {0} = label; {1} = output".format(r_label.shape, self.activations[-1].shape))
        #get the error of the last vec
        #get the error of the previous vec
        #get the gradient, save as d_Bias
        #gradient times other shit = save as d_weights
        #apply the deltas

        errors = []
        delta_weights = []
        delta_biases = []
        error = self.activations[-1] - r_label
        errors.append(error)
        
        #get all error vecs first
        for i in range(len(self.weights)-1, -1, -1):
            error = self.weights[i].T @ error
            errors.append(error.copy())
        errors.reverse()
            
        for i in range(len(self.weights)-1, -1, -1):
            #get the deltas
            derivative = np.apply_over_axes(self.activation_function.derivative, self.activations[i+1].copy(), [0,self.activations[i+1].shape[0]])
            gradient = derivative * error[i+1].T * self.learning_rate
            delta_biases.append(gradient.copy())
            delta_weights.append(gradient @ self.activations[i].copy().T)
        
        delta_weights.reverse()
        delta_biases.reverse()
        return (delta_weights, delta_biases)

    def train(self, training_data:tuple, test_data:tuple, epochs):
        input_data, label_data = training_data
        t_input_data, t_label_data = test_data
        
        e = 0
        while e < epochs:
            e += 1
            i = 0
            while i < len(input_data)-1:
                self.feedforward(input_data[i])
                bp = self.backpropagate(label_data[i])
                self.adjust(bp)
                i += 1
            j = 0
            correct = 0
            while j < len(t_input_data)-1:
                self.feedforward(t_input_data[j])
                r_label = np.reshape(t_label_data[j], (10,1))
                out = np.argmax(self.activations[-1])
                answer = np.argmax(r_label)
                if out == answer:
                    correct += 1
                j += 1
            print("EPOCH {0}: {1} / {2} correct.".format(e, correct, len(t_input_data)))


    def can_insert_data(self, new_data:np.ndarray, existing_matrix:np.ndarray) -> bool:
        if new_data.shape == existing_matrix.shape:
            return True
        return False 
    
    def adjust(self, deltas:tuple):
        dw,db = deltas
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + dw[i]
        for i in range(len(self.biases)):
            self.biases[i] = self.biases[i] + db[i]







#############################
#   ACTIVATION FUNCTIONS    #
#############################
class ActivationFunction:
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative
        
#sqiggle between 0 and 1
Sigmoid = ActivationFunction(
    lambda x,_axis: 1/(1+ np.e**(x)),
    lambda y,_axis: y * (1-y)
)

