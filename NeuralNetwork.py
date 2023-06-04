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
        layer = input.copy()
        self.pre_activations.append(input.copy())
        for n in range(self.numb_layers - 1):
            #multiply by the weights
            layer = mx.Matrix.multiply_matrix(self.weights[n], layer)
            #add the biases
            layer.add(self.biases[n])
            #save a copy of the layer to pre-activations
            self.pre_activations.append(layer.copy())
            #squish w activation function
            layer.apply_function(self.activation_function.function)
            #save the result in activation list
            self.activations.append(layer.copy()) #activations list is 1 ahead of pre-activations
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

    def adjust_network(self, batch_data:list):
        #FIXME make the errors accumulate before subtracting them
        #batch_data is a list of tuples (input, answer)
        for input, answer in batch_data:
            self.feedforward(input)
            dw, db = self.backpropagate(answer)
            #this is still stochastic *****FIX
            for w,b in self.weights, self.biases:
                w:mx.Matrix.subtract(dw)
                b:mx.Matrix.subtract(db)
    
    def train():
        #this is the training function - complex
        pass

    def assess():
        #this is the guess with untested input
        pass

    def divide_data():
        #prepare the data in a way for the network to train on it
        pass


#####################
#   SAVING NETWORK  #
#####################
    def save(self):
        pass

    def save_exists(self):
        try:
            f = open(self.save_dir)
        except:
            return False
        return True
    
    def load(self):
        assert self.save_exists(), "save data does not exist yet, save data before reading"



###############
#   UTILITY   #
###############
    def reset_error_vecs(self):
        self.error_matrices = []

    def reset_all(self):
        self.reset_error_vecs()