import numpy as np
from .activations import activation_func, activation_derivative

class Layer:
    def __init__(self, n_in, n_out, activation, w_init='xavier'):
        self.activation=activation

        if w_init=='xavier':
            limit=np.sqrt(6/(n_in+n_out))
            self.W=np.random.uniform(-limit, limit, (n_in, n_out))
        elif w_init=='zeros':
            self.W=np.zeros((n_in, n_out))
        else:
            self.W=np.random.randn(n_in, n_out)
        self.b=np.zeros((n_out, 1))

        self.grad_W=None
        self.grad_b=None
        self.A_prev=None
        self.A=None

    def forward(self, A_prev):
        self.A_prev=A_prev
        Z=np.dot(self.W.T, self.A_prev)+self.b
        self.A=Z if self.activation=='linear' else activation_func(Z, self.activation)
        return self.A
    
    def backward(self, dA, m):
        dZ=dA if self.activation=='linear' else dA*activation_derivative(self.A, self.activation)
        self.grad_W=(1/m)*np.dot(self.A_prev, dZ.T)
        self.grad_b=(1/m)*np.sum(dZ, axis=1, keepdims=True)
        return np.dot(self.W, dZ)

