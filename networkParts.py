#TODO 


import mathLib as ml
import math
import random

class Network:
    def __init__(self, layers:list, activation_function, learning_rate:float):
        #save setup settings
        self.layers = len(layers)
        self.layer_list = layers
        self.matrices = []
        self.matrices = self.get_new_matrices()
        assert type(activation_function) == ml.ActivationFunction, "ERROR: activation function must be a mathLib ActivationFunction object"
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.error_list = []

        self.randomize_weights()

    def randomize_weights(self):
        for i in range(len(self.matrices)):
            if float((i-1)/3).is_integer:
                #weight matrix
                self.matrices[i].randomize(0, 1)
            elif float((i-2)/3).is_integer:
                #bias vector
                self.matrices[i].randomize(0, 1)

    def get_new_matrices(self) -> list:
        new_matrices = []
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
                new_matrices.append(vec)
            elif float((i-1)/3).is_integer():
                #weight matrix
                wm = ml.Matrix(self.layer_list[current_layer_index + 1], self.layer_list[current_layer_index])
                new_matrices.append(wm)
            elif float((i-2)/3).is_integer:
                #bias vector
                bm = ml.Matrix(self.layer_list[current_layer_index + 1], 1)
                new_matrices.append(bm)
        return new_matrices

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

            #get the target activation layer
            output_activations = ml.Matrix.from_list(self.matrices[i].matrix)

            #get the derivative of the activation layer (derivative incorporates the sigmoid activation squish)
            output_activations.apply_function(self.activation_function.derivative)

            # gradient.print()
            # print("--------------")
            # output_activations.print()

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

        #flip all matrix lists
        error_matrices.reverse()
        self.matrices.reverse()

        # for i in range(len(error_matrices)):
        #     error_matrices[i].print()
        # print("-----------------")
        # for i in range(len(self.error_list)):
        #     self.error_list[i].print()

        #send errors to global error variable
        self.accumulate_errors(error_matrices)
        self.alter_weights(error_matrices)


    def accumulate_errors(self, error_matrices:list):
        if not self.error_list == []:
            assert len(error_matrices) == len(self.error_list), "error list length differs from previous error list stored"
            for i in range(len(error_matrices)):
                assert type(self.error_list[i]) == ml.Matrix, "error not matrix"
                self.error_list[i].add(error_matrices[i])
        else:
            self.error_list = error_matrices

    def alter_weights(self, error_matrices:list):
        #change the weights and biases
        assert len(error_matrices) == len(self.matrices), "ERROR: matrices and errors are not same length. Backprop failure."
        for i in range(len(error_matrices)):
            
            # print("ERROR")
            # error_matrices[i].print()
            # print("MATRIX")
            # self.matrices[i].print()

            self.matrices[i].add(error_matrices[i])
        self.reset_error_vecs()

    def train(self, train_data, train_labels, epochs:int):
        self.reset_all()
        # lr = float(self.learning_rate)
        # cost = 1000

        #check answer compatibility
        # assert ml.Matrix.can_matrix(train_labels), "ERROR: training labels are not vector-able"

        for n in range(epochs):
            i = random.randrange(0, len(train_labels))
            # answer_vec = ml.Matrix.from_list(train_labels[i])
            self.feedforward(train_data[i])
            #set an appropriate learning rate
            # if cost > ml.Matrix.sum_of_squared_diff_cost(self.matrices[-1], answer_vec):
            #     #its better than it was
            #     self.learning_rate *= 0.9
            # else:
            #     #its crappier than it was
            #     self.learning_rate *= 1.1
            # print(self.learning_rate)

            self.backpropagate(train_labels[i])
            self.reset_layer_vecs()
            # print(i)
            
        self.alter_weights(self.error_list)
        # self.learning_rate = lr
        # print(self.learning_rate)

    def assess(self, test_data, test_labels):
        #clear the cache
        self.reset_all()

        #check inputed test data
        assert ml.Matrix.can_vector(test_labels), "ERROR: test data incompatible to matrixize"
        test_matrix = ml.Matrix.from_list(test_labels)

        #feedforward to get output
        self.feedforward(test_data)
        #copy and round the output vector
        output_matrix = ml.Matrix.copy(self.matrices[-1])
        for i in range(output_matrix.rows):
            output_matrix.matrix[i][0] = round(output_matrix.matrix[i][0])

        #check if the rounded vector is the test data, print and return bool
        if output_matrix.matrix == test_matrix.matrix:
            # print("PASS")
            return True
        # print("FAIL")
        return False

    def predict(self, test_data):
        self.reset_all()

        self.feedforward(test_data)
        self.print_output()

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
    
    def reset_error_vecs(self):
        self.error_matrices = []

    def reset_all(self):
        self.reset_error_vecs()
        self.reset_layer_vecs()
