#TODO refactor all this shit

#------------IMPORTS-----------------
import numpy as np


#-------------CLASSES-----------------
#####################
#   NEURAL NETWORK  #
#####################
class Network:
    def __init__(self, layers:list):
        self.numb_layers = len(layers)
        self.layers = layers
        self.cost = 0.0

        self.activation_function = Sigmoid
        self.learning_rate = 3

        self.activations = [] #these are row vectors
        self.pre_activations = []
        self.weights = [] #multidimensional matrix
        self.biases = [] #these are row vectors

        for layer in layers:
            self.activations.append(np.zeros(shape = (layer,1)))

        # rng = np.random.default_rng()
        for i in range(self.numb_layers-1):
            self.weights.append(np.ones((self.layers[i+1], self.layers[i])))
            self.biases.append(np.ones((self.layers[i+1],1)))
            # self.weights.append(rng.uniform(0,1, size=(self.layers[i+1], self.layers[i])))
            # self.biases.append(rng.uniform(0,1, size=(self.layers[i+1],1)))
    
    def feedforward(self, data:np.ndarray):
        data = data.reshape((784,1))
        #check that the array is the same size as the input layer
        #assign the array as the input layer
        #prop forward until you get an output
        #make branchoff matrices and store them in the appropriate place (PRE-ACTIVATIONS or ACTIVATIONS)
        if not self.can_insert_data(data, self.activations[0]): 
            raise Exception("Incompatible sizes: {0} = data shape, {1} = input layer shape".format(data.shape, self.activations[0].shape))
        
        layer:np.ndarray = data.copy()
        self.activations[0] = data.copy()
        propagations = self.numb_layers-1

        for i in range(propagations):
            #indices
            f_vec_i = i+1

            #multiply by weights
            layer = self.weights[i] @ layer
            #add biases
            layer += self.biases[i]
            
            #save layer
            self.pre_activations.append(layer)

            # if self.can_insert_data(layer, self.pre_activations[f_vec_i]):
            layer = self.activation_function.function(layer)
            # if self.can_insert_data(layer, self.activations[f_vec_i]):
            self.activations[f_vec_i] = layer.copy()

    def backpropagate(self, label:np.ndarray) -> tuple:
        label = np.reshape(label, (10,1))
        if not label.shape == self.activations[-1].shape: 
            raise Exception("Incompatible sizes: {0} = label; {1} = output".format(label.shape, self.activations[-1].shape))
        #get the error of the last vec
        #get the error of the previous vec
        #get the gradient, save as d_Bias
        #gradient times other shit = save as d_weights
        #apply the deltas

        errors = []
        delta_weights = []
        delta_biases = []
        error = np.subtract(self.activations[-1], label)
        self.cost = 0.0
        for i in range(error.shape[0]):
            self.cost += error[i]**2
        errors.append(error)
        
        #get all error vecs first
        for i in range(len(self.weights)-1, -1, -1):
            error = self.weights[i].T @ error
            errors.append(error.copy())
        errors.reverse()
            
        for i in range(len(self.weights)-1, -1, -1):
            #get the deltas
            derivative = self.activation_function.derivative(self.activations[i+1])
            gradient = derivative * errors[i+1] * self.learning_rate
            delta_biases.append(gradient.copy())
            dw = gradient @ self.activations[i].copy().T
            delta_weights.append(dw)
        
        delta_weights.reverse()
        delta_biases.reverse()
        return (delta_weights, delta_biases, errors)

    def train(self, epochs:int, training_data:tuple, test_data:tuple = None):
        input_data, label_data = training_data
        if test_data != None:
            t_input_data, t_label_data = test_data
        
        e = 0
        while e < epochs:
            e += 1
            i = 0
            while i < (input_data.shape[0]-1):
                self.feedforward(input_data[i])
                bp = self.backpropagate(label_data[i])
                self.clear_layer_values()
                self.adjust(bp)
                i += 1
            if test_data != None:
                j = 0
                correct = 0
                while j < (t_input_data.shape[0]-1):
                    self.feedforward(t_input_data[j])
                    out = np.argmax(self.activations[-1])
                    answer = np.argmax(t_label_data[j])
                    if out == answer:
                        correct += 1
                    j += 1
                print("EPOCH {0}: {1} / {2} correct.".format(e, correct, len(t_input_data)))
            else:
                print("EPOCH {0} complete, no test".format(e))

    def test(self, test_data:tuple):
        t_input_data, t_label_data = test_data
        j = 0
        correct = 0
        while j < (t_input_data.shape[0]-1):
            self.feedforward(t_input_data[j])
            out = np.argmax(self.activations[-1])
            answer = np.argmax(t_label_data[j])
            if out == answer:
                correct += 1
            j += 1
        print("TEST: {0} / {1} correct.".format(correct, len(t_input_data)))


    def can_insert_data(self, new_data:np.ndarray, existing_matrix:np.ndarray) -> bool:
        if new_data.shape == existing_matrix.shape:
            return True
        return False 
    
    def adjust(self, deltas:tuple):
        dw,db,_e = deltas
        #this tacks on to the back of the array, not change it
        for i in range(len(self.weights)):
            self.weights[i] = np.subtract(self.weights[i], dw[i])
        for i in range(len(self.biases)):
            self.biases[i] = np.subtract(self.biases[i], db[i])
        # print("DELTA WEIGHTS: ")
        # print(dw)
        # print("\n")
        # print("DELTA BIASES: ")
        # print(db)

    def clear_layer_values(self):
        self.activations = []
        for layer in self.layers:
            self.activations.append(np.zeros(shape = (layer,1)))


    def save(self):
        file = open("data.npy", "wt")
        for n in self.weights:
            np.save(file, n)
        for n in self.biases:
            np.save(file, n)
    
    def load(self):
        file = open("data.npy", "rt")
        w,b = np.load(file)
        print(w)
        print(b)


#############################
#   ACTIVATION FUNCTIONS    #
#############################
class ActivationFunction:
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative
        
#sqiggle between 0 and 1
Sigmoid = ActivationFunction(
    lambda x: 1/(1 + np.e**(-x)) if x.any() > -10 else x*0,
    lambda y: y * (1-y)
)

