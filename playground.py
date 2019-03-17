import numpy as np

print('\n')
print(np.array([1, 2, 3]).shape)
print(np.zeros(np.array([1, 2, 3]).shape))
print('\n')
print(np.array([[1, 2, 3], [4, 5, 6]]).shape)
print(np.zeros(np.array([[1, 2, 3], [4, 5, 6]]).shape))
print('\n')
print(np.zeros(np.array([[1, 2, 3], [4, 5, 6]]).shape[0]))
print(np.zeros(np.array([[1, 2, 3], [4, 5, 6]]).shape[1]))

print('\n')
print(np.zeros(np.array([
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ],
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
]).shape[2]))


print('\n')

print(
    np.array(
        [
            [1],
            [4],
            [7]
        ]
    ).shape
)
print('\n xxx')

print(np.array([1, 2, 3]).shape)
print(np.array([1, 2, 3]).shape[0])
print(np.zeros(np.array([1, 2, 3]).shape))
print(np.zeros(np.array([1, 2, 3]).shape[0]))

print('\n')
print(np.array([[1, 2, 3]]).shape)
print(np.array([[1, 2, 3]]).shape[0])
print(np.array([[1, 2, 3]]).shape[1])
print(np.zeros(np.array([[1, 2, 3]]).shape))
print(np.zeros(np.array([[1, 2, 3]]).shape[0]))
print(np.zeros(np.array([[1, 2, 3]]).shape[1]))

print('\n')

print(np.zeros((10, 10)))