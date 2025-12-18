import numpy as np

x = np.arange(24).reshape(4,3,2)

print(x)

matrix = [
    [
        [0, 1],
        [2, 3],
        [4, 5]
    ],
    [
        [6, 7],
        [8, 9],
        [10, 11]
    ],
    [
        [12, 10],
        [14, 15],
        [16, 17]
    ],
    [
        [18, 19],
        [20, 21],
        [22, 23]
    ]

]

array = np.array(matrix)
print(array.shape)