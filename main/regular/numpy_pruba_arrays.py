import numpy as np

one = np.array([[1, 2, 3],[4, 5, 6]])
two = np.array([[7, 8, 9],[10, 11, 12]])

# print(one.reshape((one.shape[0], one.shape[1], None)).shape)

np.ones((100, 2, one.shape[1] + two.shape[1]))

result = np.vstack((one.T, two.T))
# result = np.insert(one, two, -1, axis = 1)

print(result)

[[[1 2 3]
  [7 8 9]],
 [[4 5 6]
 [10 11 12]]]
