# NumPy Basics

## Why NumPy for ML?
- Efficient N-dimensional arrays (ndarray)
- Vectorized operations — no Python loops in inner computation
- Foundation for pandas, scikit-learn, PyTorch

## Key Concepts

### Arrays vs Lists
```python
import numpy as np

# Python list — slow loops
py_list = [1, 2, 3, 4]

# NumPy array — vectorized, memory-contiguous
arr = np.array([1, 2, 3, 4])

# Shape and dtype
arr.shape    # (4,)
arr.dtype    # int64
```

### Array Creation
```python
np.zeros((3, 4))        # 3x4 matrix of zeros
np.ones((2, 3))         # 2x3 matrix of ones
np.eye(3)               # identity matrix
np.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)    # 5 evenly spaced values
np.random.randn(3, 3)   # random normal distribution
```

### Indexing and Slicing
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr[0, 1]       # 2
arr[:, 1]       # column 1 → [2, 5]
arr[arr > 3]    # boolean indexing → [4, 5, 6]
```

### Vectorized Operations
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b       # [5, 7, 9]
a * b       # [4, 10, 18]
a ** 2      # [1, 4, 9]
np.dot(a, b)    # dot product → 32
```

### Broadcasting
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr + 10        # adds 10 to every element
arr * np.array([1, 2, 3])   # multiplies element-wise per column
```

### Useful Functions for ML
```python
np.mean(arr)
np.std(arr)
np.min(arr), np.max(arr)
np.argmax(arr)   # index of max value — used in classification
np.sum(arr, axis=0)   # column sums
np.reshape(arr, (6,))
np.transpose(arr)     # or arr.T
```

## Personal Notes
- TODO: Practice matrix multiplication for understanding weight updates in neural nets
- Broadcasting rules: dimensions are compared right to left; they must be equal or one must be 1
