import numpy as np


# Element Wise Operations:

# You can either do your own naive implementation of element wise operation OR use libraries like numpy

# Element wise operations can be massively computed in parallel ( that's the reason of using libraries
# over custom implementation )

test_data_1 = np.array([
    [-1, 2, 3],
    [1, -2, 3],
    [1, 2, -3]
])

test_data_2 = np.array([
    [-1, 2, 3],
    [1, -2, 3],
    [1, 2, -3]
])

test_data_3 = np.array([
    [-1, 2, 3, 111],
    [1, -2, 3, 222],
    [1, 2, -3, 333]
])

test_data_4 = np.array([5])

# naive relu

print('----------------------------------------')
print(' Naive Implementation:')
print('----------------------------------------')


def naive_relu(x) :
    assert len(x.shape) == 2

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[ i , j ] = max( x[ i , j ] , 0)

    return x


print(naive_relu(test_data_1))

# naive addition


def naive_addition( x,  y ) :
    assert len(x.shape) == 2
    assert x.shape == y.shape

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[ i , j ] =  x[ i, j ] + y[ i , j ]

    return x


print(naive_addition( test_data_1 , test_data_2))

# Using numpy

print('----------------------------------------')
print(' Using numpy :')
print('----------------------------------------')

numpyResult = test_data_1 + test_data_2
print( numpyResult )

print( np.maximum( numpyResult , 0 ))

print('----------------------------------------')
print(' Broadcast:')
print('----------------------------------------')

# Our naive implementation as well as numpy will only work if the shape of both tensors are same

# TRY THIS : print( test_data_1 + test_data_3 )
# The above will give error : ValueError: operands could not be broadcast together with shapes (3,3) (3,4)

# But this will work
print( test_data_1 + test_data_4 )

# Other examples
print('\n')

# scaler value 5 strected and broadcasted so that virtual shape of it becomes equal to test_data_1
print( test_data_1 * 5 )

print('----------------------------------------')
print(' Dot Operation:')
print('----------------------------------------')

print( test_data_1 )
print( test_data_2 )
print('\n')

print( np.dot(  test_data_1 , test_data_2 ) )
print('\n')

# The above is different from doing test_data_1 * test_data_2
print( test_data_1 * test_data_2 )



# DOT product can be taken between

# For linear algebra and in depth of of dot product > Watch : https://www.youtube.com/watch?v=LyGKycYT2v0 (

# a. 2 vectors ( produces a scalar )
    # Rules :
    # Both vectors need to have same number of elements
# b. a matrix 'x' and a vector 'y'( produces a vector where coefficients are the dot products between y and the rows of x )
    # Rules :
    # First dimension of 'x' must be the same as 0th dimension of 'y'
# c. 2 matrices ( produces another matrix )
     # Rules :
     #  - column of first should be equal to row of second
     #  - result be a matrix with rows of first and columns of second


print('\n * Vector Dot \n')


def vector_dot(x,y):
    assert len(x.shape) == 1 # Checking if x is a vector , in numpy vector is a single row of numbers
    assert len(y.shape) == 1 # Checking if y is a vector , in numpy vector is a single row of numbers
    assert x.shape[0] == y.shape[0]

    z = 0

    for i in range( x.shape[0] ):
        z = z + ( x[i] * y[i])

    return z


print(vector_dot(np.array([2, 4]), np.array([6, 8])))
# print( vector_dot( np.array([ [2],[4] ]) , np.array([ [6],[8] ]) ) ) # <<<<<<<< THIS WILL FAIL

print('\n * Matrix Vector Dot \n')


def matrix_vector_dot(x, y):
    assert len(x.shape) == 2  # Checking if x is a matrix, in numpy matrix will have shape 2
    assert len(y.shape) == 1  # Checking if y is a vector , in numpy vector is a single row of numbers

    assert x.shape[1] == y.shape[0]  # First dimension of 'x' must be the same as 0th dimension of 'y'

    z = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] = z[i] + (x[i, j] * y[j])

    return z


print(matrix_vector_dot(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), np.array([5, 5, 5])))


print(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])[1, :])


print('\n * Matrix Matrix Dot \n')


def matrix_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2

    assert x.shape[1] == y.shape[0]

    z = np.zeros((x.shape[0], y.shape[1]))
    # Notice the double () in above statement >> https://stackoverflow.com/questions/5446522/data-type-not-understood

    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = vector_dot(row_x, column_y)
    return z


print(matrix_matrix_dot(
    np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
    ),
    np.array(
        [
            [1],
            [2],
            [3]
        ]
    )
))


# Tensor Reshaping

print('----------------------------------------')
print(' Tensor Reshaping ')
print('----------------------------------------')

sample_data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

print('\n Actual Shape: \n')
print(sample_data.shape)

# print(sample_data.reshape(5, 2))  # WILL FAIL : ValueError: cannot reshape array of size 12 into shape (5,2)

print('\n Reshaping to 3,4: \n')
print(sample_data.reshape(3, 4))
print('\n Reshaping to 6,2: \n')
print(sample_data.reshape(6, 2))
print('\n Reshaping to 12,1: \n')
print(sample_data.reshape(12, 1))

print('\n\n Using Builtin Transpose \n\n')

print('\nFrom:\n')
print( np.zeros((2, 4)))
print('\nTo:\n')
print( np.transpose( np.zeros((2, 4))))



