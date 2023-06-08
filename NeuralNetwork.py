#TODO 
#verify the creation and randomization algorithms
#should i create a input layer to save to input data? Or just start with the hidden layers? 
#-> NO must know to calculate the weight matrix shape

#TODO:
#def train -> call chop up data, save function?, 
#def chop up data -> call apply changes w batch passed in, optional evaluate.
#  chop data into train,test,dev sets 60,20,20.
#  dev set used to change learning rate, other hyperparameters.
#def apply the changes -> call feedforward, then backprop w single instance of training passed in
#def evaluate (for test data only); dev set must incorporate during training. Stagger dev sets?
#  3 sets in a row: baseline, higher, lower, then adjust learning rate according
#  to furthest difference in cost function?
# maybe not store the activations, but just the pre-activations for backprop?
#  NO we need the activations for the cost function for each layer, yes?

#LIBRARIES
import MatrixMath as mx
import random
import math

#####################
#   NETWORK CORE    #
#####################
#-------------------DIAGRAM------------------
#
#LAYER              0   1   2   3   4   5
#PRE-ACTIVATIONS    x   x   x   x   x   x
#ACTIVATIONS            x   x   x   x   x
#WEIGHTS              x   x   x   x   x
#BIASES               x   x   x   x   x
#ERRORS                 x   x   x   x   x
#D_WEIGHTS            x   x   x   x   x
#D_BIASES             x   x   x   x   x

class Network:
    def __init__(self, layers:list):
        #savefile stuff
        self.save_dir = "neural-network-data.txt"

        #arguments supplied
        self.numb_layers = len(layers)
        self.layers = layers

        #core lists
        self.pre_activations = [] #this will be identical shape-wise to self.activations
        self.activations = [] #this will be a list of the squished values in vectors
        self.weights = [] #this will be a list of the weight matrices between each vector
        self.biases = [] #this will be a list of the biases' weights between each vector (minus input)

        #secondary properties
        self.activation_function = mx.Sigmoid
        self.learning_rate = 0.1

        #initialize weights, biases
        #p = previous layer
        #n = next layer
        for p, n in zip(self.layers[:-1], self.layers[1:]): #each same length bc one index is cut each
            self.weights.append(mx.Matrix(n,p))
            self.biases.append(mx.Matrix(n,1))
        
        #randomize all matrices in weights and biases
        for matrix in self.weights:
            assert type(matrix) == mx.Matrix, "item is not a matrix in self.weights"
            matrix.randomize(-1, 1)
        for matrix in self.biases:
            assert type(matrix) == mx.Matrix, "item is not a matrix in self.biases"
            matrix.randomize(-1, 1)

#-------------CORE FUNCTIONS------------
    def feedforward(self, input:mx.Matrix) -> mx.Matrix:
        #"layer" is the main matrix object that will be manipulated, 
        # and its states will be saved as appended, copied matrix objects
        layer = mx.Matrix.copy(input)
        self.pre_activations.append(mx.Matrix.copy(input))
        for n in range(self.numb_layers - 1):
            #multiply by the weights
            layer = mx.Matrix.multiply_matrix(self.weights[n], layer)
            #add the biases
            layer.add(self.biases[n])
            #save a copy of the layer to pre-activations
            self.pre_activations.append(mx.Matrix.copy(layer))
            #squish w activation function
            layer.apply_function(self.activation_function.function)
            #save the result in activation list
            self.activations.append(mx.Matrix.copy(layer)) #activations list is 1 ahead of pre-activations
        return layer
    
    def backpropagate(self, target_data:mx.Matrix) -> tuple:
        errors = []
        delta_weights = []
        delta_biases = []
        #get all vector errors
        #set the first error vector
        error:mx.Matrix = mx.Matrix.subtract(self.activations[-1] - target_data)
        errors.append(error)
        for i in range(len(self.activations) - 1, 0, -1):
            #transpose the weight in front of the target layer
            trans_weights:mx.Matrix = mx.Matrix.transpose_matrix(self.weights[i+1])
            #multiply with the forward layer error
            error.multiply(trans_weights)
            errors.append(error)

        #get all delta weights and biases
        for i in range(len(errors)-1, -1, -1):
            #lr * E * S' * Ht
            #get the derivative *only works for sigmoid*
            derivative:mx.Matrix = mx.Matrix.apply_function(self.activations[i], self.activation_function.derivative)
            #multiply with the errors element-wise
            gradient:mx.Matrix = mx.Matrix.multiply_matrix(errors[i],derivative, True)
            #multiply learning rate element-wise with gradient
            gradient.multiply(self.learning_rate, True)

            #save as bias delta
            delta_biases.append(gradient)

            #multiply gradient with transposed activation vector (next layer)
            target_trans_matrix:mx.Matrix = self.pre_activations[i]
            if i > 0:   #reassign if not at the end
                target_trans_matrix = self.activations[i-1]
            trans_vec:mx.Matrix = mx.Matrix.transpose_matrix(target_trans_matrix)
            gradient.multiply(trans_vec)
            
            #save as weight delta
            delta_weights.append(gradient)

            #reverse the deltas to align with the other lists
            delta_biases.reverse()
            delta_weights.reverse()

            #return as a tuple
            return (delta_weights, delta_biases)

    def adjust_network(self, delta_weights, delta_biases):
        for i in range(len(delta_weights)): #both deltas should always be the same length
            self.weights[i] = mx.Matrix.add_matrix(self.weights[i], delta_weights[i])
            self.biases[i] = mx.Matrix.add_matrix(self.biases[i], delta_biases[i])

    def train(self, data_set:list):
        #set is a list of 2 lists of matrices: (inputs, answers)
        #whole set of training pairs
        input_data, label_data = data_set
        for i in range(len(input_data)):
            _output = self.feedforward(input_data[i])
            delta_weight, delta_bias = self.backpropagate(label_data[i])
            #stochastic gradient descent (immediate)
            self.adjust_network(delta_weight, delta_bias)
    
    def test(self, data_set:list) -> bool:
        #set is a list of 2 lists of matrices: (input, answer)
        #whole set of test pairs
        input_data, label_data = data
        for i in range(len(input_data)):
            output = self.feedforward(input_data[i])
            if output == label_data[i]:
                return True
            else:
                delta_weight, delta_bias = self.backpropagate(label_data[i])
                #stochastic gradient descent (immediate)
                self.adjust_network(delta_weight, delta_bias)
        return False
 

#####################
#   SAVING NETWORK  #
#####################
#STRUCTURE OF SAVE DOCUMENT:
#REPEAT FOLLOWING FOR EACH MATRIX TYPE >>>
#first line = type of matrix (weight, bias)
#second line = number of matrices
#   REPEAT FOLLOWING FOR EACH MATRIX >>>
#   first line = number of rows in the matrix 
#       (use to read the next x number of lines in the file)
#   x lines = row of matrix data

    def save(self):
        file = open(self.save_dir, "wt")
        #WEIGHT MATRICES
        #write type and number of matrices for reading
        file.writelines([str(len(self.weights)), "\n"])
        for wt in self.weights:
            #specify datatype
            wt:mx.Matrix
            #write the number of rows to use for reading
            file.writelines([str(wt.rows), "\n"])
            #write each row of data in a new line
            for row in range(wt.rows):
                file.writelines([str(wt.matrix[row]), "\n"])

        #BIAS MATRICES
        #write type and number of matrices for reading
        file.writelines([str(len(self.biases)), "\n"])
        for b in self.biases:
            #specify datatype
            b:mx.Matrix
            #write the number of rows to use for reading
            file.writelines([str(b.rows), "\n"])
            #write each row of data in a new line
            for row in range(b.rows):
                file.writelines([str(b.matrix[row]), "\n"])
        file.close()

    def save_exists(self) -> bool:
        try:
            f = open(self.save_dir)
        except:
            return False
        return True
    
    def load(self) -> tuple:
        #what will be returned at the end
        weights = []
        biases = []

        file = open(self.save_dir, "rt")
        lines:list = file.readlines()
        numb_matrices = int(lines[0])

        line_index = 1
        #WEIGHTS
        for i in range(numb_matrices):
            #get rows
            rows = int(lines[line_index])
            line_index += 1
            matrix = []
            #for each row, append data
            for row in range(rows):
                matrix.append(lines[line_index])
                line_index += 1
            #make the matrix data into a matrix, and append
            weights.append(mx.Matrix.from_list(matrix))
        
        #BIASES
        numb_matrices = int(lines[line_index])
        line_index += 1
        for i in range(numb_matrices):
            #get rows
            rows = int(lines[line_index])
            line_index += 1
            matrix = []
            #for each row, append data
            for row in range(rows):
                matrix.append(lines[line_index])
                line_index += 1
            #make the matrix data into a matrix, and append
            biases.append(mx.Matrix.from_list(matrix))
        file.close()
        return (weights, biases)
    
    def clear_save(self):
        file = open(self.save_dir, "wt")
        file.write("")
        file.close()


###############
#   UTILITY   #
###############
    def reset_error_vecs(self):
        self.error_matrices = []

    def reset_all(self):
        self.reset_error_vecs()
