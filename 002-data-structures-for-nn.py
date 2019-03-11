import numpy as np

print('-------------------')
print(' Scalar :')
print('-------------------')

a = 3

print(a)


print('-------------------')
print(' 1D Vectors :')
print('-------------------')


x = np.array([1,2,3,4,5])


print(x)
print(x.ndim)
print(x.shape)
print(x.dtype)


print('-------------------')
print(' 2D Vectors :')
print('-------------------')

y = np.array(
    [
        [1,3,5,7,9],
        [2,4,6,8,0],
        [1,2,3,4,5]
    ]
)

print(y)
print(y.ndim)
print(y.shape)
print(y.dtype)

print('-------------------')
print(' 3D Vectors :')
print('-------------------')

z = np.array([
        [
            [1,3,5,7,9],
            [2,4,6,8,0],
            [1,2,3,4,5]
        ],
        [
            [1,3,5,7,9],
            [2,4,6,8,0],
            [1,2,3,4,5]
        ],
        [
            [1,3,5,7,9],
            [2,4,6,8,0],
            [1,2,3,4,5]
        ]
    ]
)

print(z)
print(z.ndim)
print(z.shape)
print(z.dtype)

print('-------------------')
print(' Slicing Vector : 1D')
print('-------------------')

xxx = [1,2,3,4,5,6,7,8,9,10]
print( xxx )
print('\n')
print( xxx[4:8])
print( xxx[:8])
print( xxx[9:])
print( xxx[:])

print('-------------------')
print(' Slicing Vector : 2D')
print('-------------------')

yyy = np.array(
    [
        [1,3,5,7,9],
        [2,4,6,8,0],
        [1,2,3,4,5],
        [5,5,5,5,5],
        [8,8,8,8,8],
        [9,9,9,9,9],
        [0,0,0,0,0],
        [3,3,3,3,3],
        [5,5,5,5,5],
        [7,7,7,7,7]
    ]
)

print(yyy)
print('\n')
print( yyy[3:] )
print('\n')
print( yyy[3:8] )
print('\n')
print( yyy[:3])
print('\n')
print( yyy[7:])

print('-------------------')
print(' Slicing Vector : 3D')
print('-------------------')

zzz = np.array([
        [
            [1,1,1],
            [1,0,1],
            [1,1,0]
        ],
        [
            [2,2,2],
            [2,0,2],
            [2,2,0]
        ],
        [
            [3,3,3],
            [3,0,3],
            [3,3,0]
        ],
        [
            [4,4,4],
            [4,0,4],
            [4,4,0]
        ],
        [
            [5,5,5],
            [5,0,5],
            [5,5,0]
        ],
        [
            [6,6,6],
            [6,0,6],
            [6,6,0]
        ]
    ])

print( zzz.shape )
print('\n')
print( zzz[4:] )
print( zzz[4:].shape )
print('\n')
print( zzz[3:5,:,:] ) # pick starting item 4 till item 5, for each pick all rows and columns
print( zzz[3:5,:,:].shape )
print('\n')
print( zzz[:,1:,1:] ) # pick all, for each row & column, pick from start to < 2
print( zzz[:,1:,1:].shape )
print('\n')
print( zzz[:,:2,:2] ) # pick all, for each row & column, pick from start to < 2
print( zzz[:,:2,:2].shape )