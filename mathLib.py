import random
import math

#############################
#   ACTIVATION FUNCTIONS    #
#############################
class ActivationFunction:
    def __init__(self, function, reverse, derivative):
        self.function = function
        self.reverse = reverse
        self.derivative = derivative
        
#sqiggle between 0 and 1
Sigmoid = ActivationFunction(
    lambda x: 1/(1+ math.e**(-x)),
    lambda y: 0.5 if (y == 0) else (1 if (y == 1) else -1 * math.log((1/y - 1))),
    lambda y: y * (1-y)
)

#TODO finish
#diagonal between 0 and +INF
Relu = ActivationFunction(
    lambda x: max(0, x),
    lambda x: x + 1,
    lambda x: x + 1,
)

#TODO finish
#expanded sigmoid from -1 to 1
TanH = ActivationFunction(
    lambda x: (e**x - e**(-x)) / (e**x + e**(-x)),
    lambda x: x + 1,
    lambda x: x + 1
)

###############
#   CLASSES   #
###############


class Matrix:
    #21apr/23 - good
    def __init__(self, rows:int, columns:int):
        self.rows = rows
        self.columns = columns
        self.matrix = []

        for y in range(rows):
            row = [0] * columns
            self.matrix.append(row)

    #21apr/23 - good
    def print(self):
        Matrix.print_matrix(self)

    #21apr/23 - good
    def randomize(self, min:int, max:int):
        for row in range(self.rows):
            for col in range(self.columns):
                cell = random.uniform(min , max)
                self.matrix[row][col] = cell

    #21apr/23 - good
    def add(self, input):
        #add matrix
        if type(input) == Matrix:
            #check if matrices are compatible
            assert Matrix.is_same_dimensions(self, input), "ERROR: Input matrix not compatible to add"

            #add matrices
            for row in range(self.rows):
                for col in range(self.columns):
                    self.matrix[row][col] += input.matrix[row][col]

        #add scalar
        elif type(input) == int or type(input) == float:
            for row in range(self.rows):
                for col in range(self.columns):
                    self.matrix[row][col] += input

        #any other input type
        else:
            print("ERROR: Invalid input to add")

    #21apr/23 - good
    def subtract(self, input):
        #subtract matrix
        if type(input) == Matrix:
            #check if matrices are compatible
            assert Matrix.is_same_dimensions(self, input), "ERROR: Input matrix not compatible to subtract"

            #subtract matrices
            for row in range(self.rows):
                for col in range(self.columns):
                    self.matrix[row][col] -= input.matrix[row][col]

        #subtract scalar
        elif type(input) == int or type(input) == float:
            for row in range(self.rows):
                for col in range(self.columns):
                    self.matrix[row][col] -= input

        #any other input type
        else:
            print("ERROR: Invalid input to subtract")

    #21apr/23 - good
    def multiply(self, input, by_element:bool = False):
        new_matrix = []
        #check matrix
        if type(input) == Matrix:

            #element-wise multiplication
            if by_element:
                assert Matrix.is_same_dimensions(self, input), "ERROR: input is not same size; unable to multiply element-wise"
                for row in range(self.rows):
                    for col in range(self.columns):
                        self.matrix[row][col] *= input.matrix[row][col]
                return

            else:
                #ensure matrices are compatible
                assert Matrix.can_multiply_matrices(self, input), "ERROR: Incompatible sizes of matrices to multiply"
                
                #get new matrix dimensions
                new_rows = len(self.matrix)
                new_cols = len(input.matrix[0])
                #for looping purposes, get the common dimension
                comp_numb = len(input.matrix)
                
                #make new matrix
                for row in range(new_rows):
                    new_row = []
                    for col in range(new_cols):
                        #each cell of new matrix will be the sum-product of row-col associations.
                        #each sum-product will have N terms; N = common dimension
                        #(row of first * col of second) with reciprocal dimension + by N
                        sum_prod = 0
                        for i in range(comp_numb):
                            sum_prod += (self.matrix[row][i] * input.matrix[i][col])
                        new_row.append(sum_prod)
                    new_matrix.append(new_row)
                
                #assign new data to self
                self.rows = new_rows
                self.columns = new_cols
                self.matrix = new_matrix

        #multiply scalar
        elif type(input) == int or type(input) == float:
            for row in range(self.rows):
                for col in range(self.columns):
                    self.matrix[row][col] *= input

        #any other input type
        else:
            print("ERROR: Invalid input to multiply")

    #21apr/23 - good
    def apply_function(self, func):
        for row in range(self.rows):
            for col in range(self.columns):
                self.matrix[row][col] = func(self.matrix[row][col])

    #21apr/23 - good
    def transpose(self):
        #make new matrix
        new_matrix = []
        for col in range(len(self.matrix[0])):
            new_row = []
            for row in range(len(self.matrix)):
                new_row.append(self.matrix[row][col])
            new_matrix.append(new_row)
        #swap rows and columns, input new matrix data
        self.rows, self.columns = self.columns, self.rows
        self.matrix = new_matrix

    #21apr/23 - good
    def clamp(self, min, max):
        assert max > min, "ERROR:MAX UNDER MIN"
        for row in range(self.rows):
            for col in range(self.columns):
                if self.matrix[row][col] < min:
                    self.matrix[row][col] = min
                if self.matrix[row][col] > max:
                    self.matrix[row][col] = max

    #####################
    #   STATIC METHODS  #
    #####################
    
    #21apr/23 - good
    @staticmethod
    def from_list(input:list, is_vector_row:bool=True):
        #check type
        assert type(input) == list, "ERROR: Invalid input, not a list"

        #initial setup
        new_rows = 0
        new_columns = 0
        new_matrix = []
        matrix_obj = Matrix(1,1)

        #decide what the input is
        #matrix
        if Matrix.can_matrix(input):
            #check that the is_vector_row argument is not changed from default, may mean user meant vector
            assert is_vector_row, "ALERT: input is matrix. Did you mean to input a vector?"
            #set new dimensions
            new_rows = len(input)
            new_columns = len(input[0])
            #make each row new, append
            for row in range(new_rows):
                new_matrix.append(list(input[row]))

        #vector
        elif Matrix.can_vector(input):
            #determine direction of vector
            #row vector
            if is_vector_row:
                #set new dimensions
                new_rows = 1
                new_columns = len(input)
                new_matrix.append(list(input))

            #column vector
            else:
                #set new dimensions
                new_rows = len(input)
                new_columns = 1
                for i in input:
                    new_matrix.append([i])

        #neither
        else:
            print("ERROR: input is invalid to convert to Matrix")
            return
        
        #assign new data and return
        matrix_obj.rows = new_rows
        matrix_obj.columns = new_columns
        matrix_obj.matrix = new_matrix
        return matrix_obj

    #21apr/23 - good
    @staticmethod
    def add_matrix(matrix, input):
        #checks
        assert type(matrix) == Matrix, "ERROR: matrix is not a matrix"
        #make copy of matrix
        c_matrix = Matrix.copy(matrix)
        
        #initial setup
        new_rows = matrix.rows
        new_columns = matrix.columns
        new_matrix = []
        matrix_obj = Matrix(1,1)
        
        #Check type of input
        #add as matrix
        if type(input) == Matrix:
            #make copy of input matrix
            c_input = Matrix.copy(input)
            #check if matrices are compatible
            assert Matrix.is_same_dimensions(c_matrix, input), "ERROR: Input matrix not compatible to add"
            
            #add matrices
            for row in range(c_matrix.rows):
                new_row_data = []
                for col in range(c_matrix.columns):
                    new_row_data.append(c_matrix.matrix[row][col] + c_input.matrix[row][col])
                new_matrix.append(new_row_data)

        #add as scalar
        elif type(input) == int or type(input) == float:
            #copy input
            c_input = float(input)

            for row in range(c_matrix.rows):
                new_row = []
                for col in range(c_matrix.columns):
                    new_row.append(c_matrix.matrix[row][col] + c_input)
                new_matrix.append(new_row)

        #any other input type
        else:
            print("ERROR: Invalid input to add")
            return
        
        #make the new object, and return it    
        matrix_obj.rows = new_rows
        matrix_obj.columns = new_columns
        matrix_obj.matrix = new_matrix
        return matrix_obj
    
    #21apr/23 - good
    @staticmethod
    def subtract_matrix(matrix, input):
        #checks
        assert type(matrix) == Matrix, "ERROR: matrix is not a matrix"
        #make copy of matrix
        c_matrix = Matrix.copy(matrix)
        
        #initial setup
        new_rows = matrix.rows
        new_columns = matrix.columns
        new_matrix = []
        matrix_obj = Matrix(1,1)
        
        #Check type of input
        #subtract as matrix
        if type(input) == Matrix:
            #make copy of input matrix
            c_input = Matrix.copy(input)
            #check if matrices are compatible
            assert Matrix.is_same_dimensions(c_matrix, input), "ERROR: Input matrix not compatible to subtract"
            
            #subtract matrices
            for row in range(c_matrix.rows):
                new_row_data = []
                for col in range(c_matrix.columns):
                    new_row_data.append(c_matrix.matrix[row][col] - c_input.matrix[row][col])
                new_matrix.append(new_row_data)

        #subtract as scalar
        elif type(input) == int or type(input) == float:
            #copy input
            c_input = float(input)
            
            for row in range(c_matrix.rows):
                new_row = []
                for col in range(c_matrix.columns):
                    new_row.append(c_matrix.matrix[row][col] - c_input)
                new_matrix.append(new_row)

        #any other input type
        else:
            print("ERROR: Invalid input to subtract")
            return
        
        #make the new object, and return it    
        matrix_obj.rows = new_rows
        matrix_obj.columns = new_columns
        matrix_obj.matrix = new_matrix
        return matrix_obj

    #21apr/23 - good
    @staticmethod
    def multiply_matrix(matrix, input, by_element:bool = False):
        #checks
        assert type(matrix) == Matrix, "ERROR: matrix is not a matrix"
        #make copy of matrix
        c_matrix = Matrix.copy(matrix)
        
        #initial setup
        new_rows = c_matrix.rows
        new_columns = c_matrix.columns
        new_matrix = []
        matrix_obj = Matrix(1,1)

        #check input type
        #multiply as matrix
        if type(input) == Matrix:
            #copy input matrix
            c_input = Matrix.copy(input)
            #element-wise multiplication
            if by_element:
                #check compatible dimensions
                assert Matrix.is_same_dimensions(c_matrix, c_input), "ERROR: input is not same size; unable to multiply element-wise"

                #do element-wise multiplication of same-dimension matrices
                for row in range(c_matrix.rows):
                    new_row = []
                    for col in range(c_matrix.columns):
                        new_row.append(c_matrix.matrix[row][col] * c_input.matrix[row][col])
                    new_matrix.append(new_row)

            else:
                #ensure matrices are compatible
                assert Matrix.can_multiply_matrices(c_matrix, input), "ERROR: incompatible sizes of matrices to multiply"

                #get new matrix dimensions
                new_rows = len(c_matrix.matrix)
                new_columns = len(c_input.matrix[0])
                #for looping purposes, get the common dimension
                comp_numb = len(c_input.matrix)

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

        #multiply as scalar
        elif type(input) == int or type(input) == float:
            #copy input
            c_input = float(input)

            for row in range(c_matrix.rows):
                new_row = []
                for col in range(c_matrix.columns):
                    new_row.append(c_matrix.matrix[row][col] * c_input)
                new_matrix.append(new_row)

        #any other input type
        else:
            print("ERROR: Invalid input to add")
            return

        #make the new object, and return it    
        matrix_obj.rows = new_rows
        matrix_obj.columns = new_columns
        matrix_obj.matrix = new_matrix
        return matrix_obj
    
    #21apr/23 - good
    @staticmethod
    def transpose_matrix(matrix):
        #ensure matrix integrity
        assert type(matrix) == Matrix, "ERROR: Not a matrix"
        #copy matrix
        c_matrix = Matrix.copy(matrix)

        #initial setup
        new_rows = c_matrix.columns
        new_columns = c_matrix.rows
        new_matrix = []
        matrix_obj = Matrix(1,1)

        for col in range(c_matrix.columns):
            new_row = []
            for row in range(c_matrix.rows):
                new_row.append(matrix.matrix[row][col])
            new_matrix.append(new_row)

        #return new matrix object
        matrix_obj.rows = new_rows
        matrix_obj.columns = new_columns
        matrix_obj.matrix = new_matrix
        return matrix_obj

    #21apr/23 - good
    @staticmethod
    def from_map(matrix, func):
        #checks
        assert type(matrix) == Matrix, "ERROR: matrix is not matrix"
        #copy
        c_matrix = Matrix.copy(matrix)

        #initial setup
        new_rows = c_matrix.rows
        new_columns = c_matrix.columns
        new_matrix = []
        matrix_obj = Matrix(1,1)

        for row in range(c_matrix.rows):
            new_row = []
            for col in range(c_matrix.columns):
                new_value = func(c_matrix.matrix[row][col])
                new_row.append(new_value)
            new_matrix.append(new_row)

        #assign new values and return new matrix object
        matrix_obj.rows = new_rows
        matrix_obj.columns = new_columns
        matrix_obj.matrix = new_matrix
        return matrix_obj

    #21apr/23 - good
    @staticmethod
    def get_row_percentage(matrix):
        #checks
        assert type(matrix) == Matrix, "ERROR: Input is not a matrix"
        #copy
        c_matrix = Matrix.copy(matrix)

        #intial setup
        new_rows = c_matrix.rows
        new_columns = c_matrix.columns
        new_matrix = []
        matrix_obj = Matrix(1,1)

        #get a list of row sums
        row_sums_list = []
        for row in range(new_rows):
            row_sum = 0
            for col in range(new_columns):
                row_sum += c_matrix.matrix[row][col]
            row_sums_list.append(row_sum)

        #make new matrix with pergentaged values
        for row in range(new_rows):
            new_row = []
            for col in range(new_columns):
                new_row.append(c_matrix.matrix[row][col] / row_sums_list[row])
            new_matrix.append(new_row)

        #return new matrix obj
        matrix_obj.rows = new_rows
        matrix_obj.columns = new_columns
        matrix_obj.matrix = new_matrix
        return matrix_obj

    #21apr/23 - good
    @staticmethod
    def copy(matrix):
        #check if matrix
        assert type(matrix) == Matrix, "ERROR: not a matrix; cannot copy"
        new_matrix = []

        #if a 2d matrix
        if type(matrix.matrix[0]) == list:
            for row in range(matrix.rows):
                    new_row = list(matrix.matrix[row])
                    new_matrix.append(new_row)
        #if a column vector
        elif type(matrix.matrix[0]) == float:
            for row in range(matrix.rows):
                new_row = float(matrix.matrix[row])
                new_matrix.append(new_row)
        
        matrix_obj = Matrix(matrix.rows, matrix.columns)
        matrix_obj.matrix = new_matrix
        return matrix_obj

    #####################
    #   COST FUNCTIONS  #
    #####################
    #FIXME BROKEN
    @staticmethod
    def sum_of_squared_diff_cost(output_vector:list, target_vector:list) -> float:
        cost = 0.0
        vec_diff = Matrix.subtract_matrix(output_vector, target_vector)
        for row in range(len(vec_diff)):
            value = vec_diff[row] * vec_diff[row]
            cost += value
        return cost

    #########################################
    #   METHODS TO CHECK MATRIX INTEGRITY   #
    #########################################
    
    #21apr/23 - good
    @staticmethod
    def can_multiply_matrices(matrix1, matrix2) -> bool:
        #check if matrices
        if not type(matrix1) == Matrix: return False
        if not type(matrix2) == Matrix: return False
        #get comparable dimensions
        m1_cols = len(matrix1.matrix[0])
        m2_rows = len(matrix2.matrix)
        #check if comparable dimensions are same length
        if not m1_cols == m2_rows: return False
        return True
    
    #21apr/23 - good
    @staticmethod
    def is_same_dimensions(matrix1, matrix2) -> bool:
        if not type(matrix1) == Matrix: return False
        if not type(matrix2) == Matrix: return False
        if not len(matrix2.matrix) == len(matrix1.matrix): return False
        for y in range(len(matrix2.matrix)):
            if not type(matrix1.matrix[y]) is list: return False
            if not type(matrix2.matrix[y]) is list: return False
            if not len(matrix2.matrix[y]) == len(matrix1.matrix[y]): return False
            for x in range(len(matrix2.matrix[y])):
                if type(matrix1.matrix[y][x]) is list: return False
                if type(matrix2.matrix[y][x]) is list: return False
        return True
    
    #21apr/23 - good
    @staticmethod
    def is_same_dimensions_detailed(matrix1, matrix2) -> str:
        #checking objects
        if not type(matrix1) == Matrix: print("ERROR: matrix1 is not a matrix")
        if not type(matrix2) == Matrix: print("ERROR: matrix2 is not a matrix")
        if not len(matrix2.matrix) == len(matrix1.matrix): print("ERROR: rows not compatible")
        #checking contents of matrices - rows
        for y in range(len(matrix2.matrix)):
            if not type(matrix1.matrix[y]) is list: return "ERROR: matrix1 less than 2d list as input at row " + str(y)
            if not type(matrix2.matrix[y]) is list: return "ERROR: matrix2 less than 2d list as input at row " + str(y)
            if not len(matrix2.matrix[y]) == len(matrix1.matrix[y]): return "ERROR: columns not compatible"
            #checking contents of matrices - columns
            for x in range(len(matrix2.matrix[y])):
                if type(matrix1.matrix[y][x]) is list: return "ERROR: matrix1 more than 2d list as input at row,col: (" +str(y) + "," + str(x) + ")"
                if type(matrix2.matrix[y][x]) is list: return "ERROR: matrix2 more than 2d list as input at row,col: (" +str(y) + "," + str(x) + ")"
        return "Compatible matrices confirmed"
    
    #21apr/23 - good
    @staticmethod
    def can_matrix(array:list) -> bool:
        for y in range(len(array)):
            if not type(array[y]) is list: return False
            if not len(array[0]) == len(array[y]): return False
            for x in range(len(array[y])):
                if type(array[y][x]) is list: return False
        return True

    #21apr/23 - good
    @staticmethod
    def can_vector(array:list) -> bool:
        for i in array:
            if type(i) == list: return False
        return True

    #21apr - good
    @staticmethod
    def print_matrix(new_matrix):
        assert type(new_matrix) == Matrix, "ERROR: Input is not a matrix; No matrix to print"
        output = ''
        for row in range(len(new_matrix.matrix)):
            output += '['
            for col in range(len(new_matrix.matrix[row])):
                output += str(new_matrix.matrix[row][col])
                if not col == (len(new_matrix.matrix[row]) - 1):
                    output += ','
            output += ']\n'
        print(output)


e = [1,1,1,1]
a = Matrix(2,3)
b = Matrix(2,3)
a.add(2)
b.add(3)
c = Matrix.get_row_percentage(a)
c.add(2)

a.print()
c.print()