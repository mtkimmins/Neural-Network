ERRORS:

the matrix for hidden->output weights is [0,0], while the matrix for error hidden ->  output weights is [0]
the rest seem compatible


HOW IT WORKS:

always weights * activations, not the other way around


FROM INIT
# self.matrices = {
#     "inputs":,
#     "weights0":,
#     "biases0":,
#     "layer1":,
#     "weights1":,
#     biases1:,
#     ...
#     outputs:
# }
# ^^^ This is quite bad to name indices, then get those indices accurately
#
# complete library of matrices        
#[0]    [0,0,0,0]     [0]      |     [0]     [0,0,0]         [0]       |      [0]
#[0]    [0,0,0,0]     [0]      |     [0]     [0,0,0]         [0]       |      [0]
#[0]    [0,0,0,0]     [0]      |     [0]                               |
#[0]
#input  weights i-h   biases i-h     hidden  weights h-o     biases h-o       output
#|-------this is 1 layer-------|
# weight matrix indices would be 1,4,7,10... 1+(3*N) from the input side
#   which matrix? W_numb = (i-1)/3; matrix 0 would be the weights for layer 0
# layer indices would be 0,3,6... (3*N) from the input side
#   which layer? L_numb = i/3 starting at 0 as input
# bias indices would be 2,5,8... 2+(3*N) from the input side
#   which matrix? B_numb = (i-2)/3; matrix 0 would be the biases for layer 0

#amount of indices in matrices is (len(layer_list) + (len(layer_list) - 1)*2)
#then loop and append in this order: 
# activation vector (blank)
# weight matrix (random)
# bias vector (random)
#translate index from matrices to layer list
#   floor(i/3)
#DO NOT APPLY THE ACTIVATION FUNCTION HERE, only do that as a separate matrix for calculation in the next round

FROM BACKPROP:
#I need to remember what the activations of the feedforward were
        #
        # ERRORS
        # this is a new list of matrices
        #[0]    [0]         [0,0,0]     [0]      [0]        [0,0,0,0]
        #[0]    [0]         [0,0,0]     [0]      [0]        [0,0,0,0]
        #                               [0]      [0]        [0,0,0,0]
        #
        #output biases o-h  weights o-h hidden   biases h-i weights h-i     *dont count input error
        #error  changes     changes     error    changes    changes
        #
        #loop i would be (len(self.matrices) - 1)
        #order of loop = 
        #first one = output - error
        #bias changes = (lr*last error vec (i-1) * activation') * self.matrices[-(1+i+2)] transposed ; just 2 layers over
        #       the corresponding matrix index is self.matrices[-(1+i)]
        #       1 for end, i for current indice away from edge, 2 for layers away = activation of left layer
        #weight changes = (lr*last error vec (i-2) * activation') * self.matrices[-(1+i+1)] transposed; just 1 layer over
        #       use the static transposer
        #repeat
        #will have a gate to check what indices are allocated to what operation
        # errors are 0,3,6...
        # biases are 1,4,7...  BIASES AND WEIGHTS ARE EFFECTIVELY SWITCHED, WHILE ERROR VECS ARE SAME -- indices
        # weights are 2,5,8...





        #get a list of all the error vectors
        #    = weight matrix transposed * error vec
        #calc gradient for weight
        #    = lr*error_vec(layer to right)*activation'
        #gradient * activation_vec(layer to left) transposed = matrix of changes desired to weights
        #change weights
        #forget error vecs
        #forget activation vecs

        #delta_w = lr * error_vec * Sigmoid' * H
        
        
        #make target vec
        #make error vec holder
        #get error by subtract the output we got from the target vec
        #multiply the last error vec with the next weight matrix, but transposed = new error vec
        #append to holder

FROM MATHLAB MATRIX MULTIPLICATION:
#do matrix multiplication
                for row in range(new_rows):
                    new_row = []
                    for col in range(new_columns):
                        #each cell of new matrix will be the sum-product of row-col associations.
                        #each sum-product will have N terms; N = common dimension
                        #(row of first * col of second) with reciprocal dimension + by N
                        sum_prod = 0
                        for j in range(comp_numb):
                            sum_prod += (c_matrix.matrix[row][j] * c_input.matrix[j][col])
                        new_row.append(sum_prod)
                    new_matrix.append(new_row)

ERRORS SEEN:

late April/23
- all outputs were 0.9; this was due to lack of deep copying the matrices. If multiplied, 
all matrices involved change. Made deep copies embedded into static methods of matrix library.

1 May 2023
-consecutively using the predict method would produce different results each time despite no intermittent
training; this was due to the lack of clearing the vectors before the following feedforward call. As the new
vectors are simply added to presumably blank vector matrices, the vector values accumulated, producing different
results each time. Used the reset_vector_values() to clear the vectors at the end of prediction like in train().

2 May 2023
-outputs were 0.01,0.5,0.5,0.9; this may be due to getting to a local min, gettings stuck. Learning rate
adjustemnts seem to change these values. A diminishing learning rate changes all outputs to the same = 0.7-0.5.
Solution could be to batch tweak the network instead of adjusting the network each feedforward/backprop cycle.
Try adding up the errors from backprop then changing the errors alltogether. Yet this may be equivalent to
adding it each cycle. Except the errors will be constant if not changed vs always different if changed each time.
Thus, these strategies will yield different results.

2 May 2023
-the first output for predict would always be different than the other inputs; this is due to forgetting to
clear the error vec memory, or the activation vector memory before calculating a brand new error. Reset these
matrices before predict and train, solved it.

2 May 2023
-if weights changed after each training session AND each epoch, then the errors of 3/4 outputs predicted will
be identically wrong, while the other will be correct. Alternatively, this double-up method seems to bring
the outputs to a definitive point of 1.0 and very low 1e-25 numbers, so that's good.

2 May 2023
-only sometimes it randomly works; ChatGPT suggested that the network is just too small as a 2,2,1. Increasing
the hidden layer size (to 5) seems to produce correct predictions 3 for 3 tries in a row. Thus, a 2,2,1 network is
possible, but too damn small for reliable recreation. UPDATE: still gets stuck.

5 May 2023
-sometimes it works and sometimes it doesnt. Im assuming that it is just due to getting stuck at a local min. This is
why its important to check how well the network is performing before prediction.