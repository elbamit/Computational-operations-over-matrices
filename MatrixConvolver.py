import numpy as np
import copy


#Helper function - multiply 2 matrices and sums every cell of the result matrix.
#Attributes - 'first' (ndarray), 'second' (ndarray)
#Return type - int.
def matrix_mult(first, second):
        result = 0
        for row in range(first.shape[0]):
            for col in range(second.shape[0]):
                result += first[row, col] * second[row, col]
        return result 
    

class MatrixConvolver:
    
    #Constructor method - creates an empty list wich will later receive arrays as objects.
    def __init__(self):
        self.matrices_list = []
        
    #Method to add a matrix (ndarray) to the matrices_list.
    #Only if shape of the matrix is identical to the shape of the matrices that already exist in list.
    def add_matrix(self,matrix):
        if len(self.matrices_list) == 0:
            self.matrices_list.append(matrix)
        else:
            if matrix.shape == self.matrices_list[0].shape:
                self.matrices_list.append(matrix)
    
    #Method to remove a matrix from the matrices_list.
    #'element' is ndarray type - if 'element' exists, removes it.
    #'element' is int type - removes the object at index 'element'
    #'element' is any other type - return '-1'.
    def remove_matrix(self, element):
        if isinstance(element, np.ndarray):#Checks if 'element' is 'ndarray' type
            for i in range(len(self.matrices_list)):
                if np.array_equal(element, self.matrices_list[i]):#Check if 'element' is in the matrices list.
                    self.matrices_list.pop(i)
                    break #if element has been removed, stops searching in others elements.
                    
        elif isinstance(element, int):#Checks if 'element' is 'int' type.
            self.matrices_list.pop(element)
        
        else:
            return -1
    
    #Returns a deepcopy of the matrices_list.
    def get_matrices(self):
        return copy.deepcopy(self.matrices_list)
    
    
    #Reshapes the matrices list to the 'new_shape' shape.
    #'new_shape' is a tuple holding two int numbers.
    def reshape_matrices(self, new_shape):
        if len(self.matrices_list) == 0: #if matrices list is empty, returns 0
            return 0
        else:
                for i in range(len(self.matrices_list)):
                    try: #Try to reshape the matrix to the given shape.
                        self.matrices_list[i] = self.matrices_list[i].reshape(new_shape)
            
                    except: #Returns -1 if it impossible to reshape the matrix to the 'new_shape'.
                        return -1
    
    
    #Convolution implementation. Returns a matrix (ndarray).
    #Attributes: 'i' (int) - represents the matrix from the matrices list on wich the convolution will be performed.
                #'filter_matrix' (ndarray) - the matrix that will be used for the convolution.
                #'stride_size' (int) default = 1 - represents the steps size 'filter_matrix' will use to move. 
    def conv(self, i, filter_matrix, stride_size=1):
        #finds the matrix on wich we want to perform the convolution
        big_matrix = self.matrices_list[i]
        
        #Gives the length of row/column for the result matrix.
        result_size = int((((big_matrix.shape[0] - filter_matrix.shape[0])/stride_size) +1))
        
        #Creates a matrix of '0' with shape of the result_size.
        cell_res = np.zeros((result_size, result_size))
        
        for i in range(result_size):
            for j in range(result_size):
                #Uses helper func 'matrix_mult' to multiply between the 2 corresponding matrices.
                #'big_matrix' is sliced in row and column to fit the 'filter_matrix'
                cell_res[(i,j)] = matrix_mult(big_matrix[i:i+filter_matrix.shape[0], j:j+filter_matrix.shape[0]], filter_matrix)    
                
        return cell_res
        