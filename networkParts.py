import mathLib as ml
import math
import random

class Network:
    def __init__(self, inputs:int, hidden_layers:list, outputs:int, activation_function, learning_rate:float):
        #save setup settings
        self.layer_list = [inputs] + hidden_layers + [outputs]
        self.matrices = []
        assert type(activation_function) == ml.ActivationFunction, "ERROR: activation function must be a mathLib ActivationFunction object"
        self.activation_function = activation_function
        self.learning_rate = learning_rate

        # complete library of matrices        
        #[0]    [0,0,0,0]     [0]      |     [0]     [0,0,0]         [0]       |      [0]
        #[0]    [0,0,0,0]     [0]      |     [0]     [0,0,0]         [0]       |      [0]
        #[0]    [0,0,0,0]     [0]      |     [0]                               |
        #[0]
        #input  weights i-h   biases i-h     hidden  weights h-o     biases h-o       output
        #|-------this is 1 layer-------|

        #set up matrices for network
        for i in range(len(self.layer_list)+ (len(self.layer_list) -1) * 2):
            current_layer_index = (math.floor(i/3))

            if float(i/3).is_integer():
                #layer vector
                vec = ml.Matrix(self.layer_list[current_layer_index], 1)
                self.matrices.append(vec)
            elif float((i-1)/3).is_integer():
                #weight matrix
                wm = ml.Matrix(self.layer_list[current_layer_index + 1], self.layer_list[current_layer_index])
                wm.randomize(0, 1)
                self.matrices.append(wm)
            elif float((i-2)/3).is_integer:
                #bias vector
                bm = ml.Matrix(self.layer_list[current_layer_index + 1], 1)
                bm.randomize(0, 1)
                self.matrices.append(bm)

    def feedforward(self, input_data):
        assert len(input_data) == self.layer_list[0], "ERROR: input data is not in compatible format for input vector"
        
        #set the input data as first layer
        self.matrices[0] = ml.Matrix.from_list(input_data, False)

        #make a loop for all layers except for the output layer (since this layer will be set by the previous layer)
        for i in range(len(self.matrices) - 1):
            #layer vectors only
            if not float(i/3).is_integer(): continue

            #for sanity's sake, the indices of the relevant components of each layer in respect to the current layer vector
            current_weight_index = (i + 1)
            current_bias_index = (i + 2)
            next_layer_index = (i + 3)

            #next matrix layer vector of activations = wt * current vector activations
            next_layer_vec = ml.Matrix.multiply_matrix(self.matrices[current_weight_index], self.matrices[i])
            #add the current layer bias
            next_layer_vec.add(self.matrices[current_bias_index])
            #since the next layer vector should be blank, just add the vector calculated above
            self.matrices[next_layer_index].add(next_layer_vec)
            #squish result with an activation function
            self.matrices[next_layer_index].apply_function(self.activation_function.function)
                
    def backpropagate(self, target_data):
        #flip the matrices to go backward
        self.matrices.reverse()

        #make target vector
        assert len(target_data) == self.layer_list[-1], "ERROR: target data is not in compatible format for output vector"
        target_vec = ml.Matrix.from_list(target_data, False)

        #make error matrices variable
        error_matrices = []
        #calc output error
        e_vec = ml.Matrix.subtract_matrix(target_vec, self.matrices[0])
        #append the first error
        error_matrices.append(e_vec)

        #loop through all layers; remember, the matrices are currently reversed
        for i in range(len(self.matrices) - 1):
            #to be appended at the end
            errors = []

            #layer vectors only
            if not float(i/3).is_integer(): continue

            #relevant indices
            bias_index = (i + 1)
            weight_index = (i + 2)
            backward_layer_index = (i + 3)

            #get all layer errors
            #current layer error = TransposedWeightMatrix * OutputErrors

            #delta weights (as described in forward self.matrices fashion)
            #delta weights = lr * OutputErrorVec * d'(output) * TransposedCurrentLayerActivations

            #backward layer vec errors
            t_wm = ml.Matrix.transpose_matrix(self.matrices[weight_index])
            t_wm_p = ml.Matrix.get_row_percentage(t_wm)
            backward_layer_error_vec = ml.Matrix.multiply_matrix(t_wm_p, error_matrices[i])
            #calculate the gradient: 
            #1) learning rate * forward error vector; 
            gradient = ml.Matrix.multiply_matrix(error_matrices[i], self.learning_rate)
            #2) gradient * derivative of activation function used on the activation of the layer


            output_activations = ml.Matrix.from_list(self.matrices[i].matrix)

            # last_vec = ml.Matrix.from_map(self.matrices[prev_layer_vec], ml.f_sigmoid)
            # derive layer pre-activations






            output_activations.apply_function(self.activation_function.derivative)




            #gradient * derived activation vector
            gradient.multiply(output_activations, True)

            #transpose the activation vector behind the index
            t_input_error_vec = ml.Matrix.transpose_matrix(self.matrices[backward_layer_index])

            #multiply the whole gradient vector by the transposed previous layer vector to make a square matrix
            wmd = ml.Matrix.multiply_matrix(gradient, t_input_error_vec)

            #append in this order: bias, weight, layer
            error_matrices.append(gradient)
            error_matrices.append(wmd)
            error_matrices.append(backward_layer_error_vec)

        #change the weights and biases
        assert len(error_matrices) == len(self.matrices), "ERROR: matrices and errors are not same length. Backprop failure."
        for i in range(len(error_matrices)):
            
            # print("ERROR")
            # error_matrices[i].print()
            # print("MATRIX")
            # self.matrices[i].print()

            self.matrices[i].add(error_matrices[i])

        #flip the matrix list back again for future feedforwards
        self.matrices.reverse()

    def train(self, train_data, train_labels, epochs:int):
        for n in range(epochs):
            i = random.randrange(0, len(train_data))
            self.feedforward(train_data[i])
            self.backpropagate(train_labels[i])
            self.reset_layer_vecs()

    def predict(self, test_data):
        self.feedforward(test_data)
        self.print_output()

#def evaluate(self, test_data, test_labels):

#def save(self, filename):

#def load(self, filename):

###############
#   UTILITY   #
###############
    def print(self):
        print(self.matrices)
        for matrix in self.matrices:
            matrix.print()

    def print_output(self):
        self.matrices[-1].print()

    def reset_layer_vecs(self):
        for layer in range(len(self.layer_list)):
            vec = ml.Matrix(self.layer_list[layer], 1)
            self.matrices[layer*3] = vec