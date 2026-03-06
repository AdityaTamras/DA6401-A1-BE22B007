import numpy as np

def activation_func(z, type):
    if type=='relu':
        return np.maximum(0, z)
    elif type=='tanh':
        return np.tanh(z)
    elif type=='sigmoid':
        return 1 / (1 + np.exp(-z))
    elif type=='softmax':
        exp_z=np.exp(z-np.max(z, axis=0, keepdims=True))
        return exp_z/np.sum(exp_z, axis=0, keepdims=True)
    else:
        raise ValueError(f"Invalid activation function: {type}")

def activation_derivative(A, type):
    if type=='relu':
        return np.where(A>0, 1, 0)
    elif type=='tanh':
        return 1 - np.power(A, 2)
    elif type=='sigmoid':
        return A*(1-A)
    else:
        return 1