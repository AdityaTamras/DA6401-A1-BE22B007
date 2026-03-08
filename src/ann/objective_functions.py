import numpy as np
from .activations import activation_derivative

def softmax(Z):
    exp_z=np.exp(Z-np.max(Z, axis=0, keepdims=True))
    return exp_z/np.sum(exp_z, axis=0, keepdims=True)

def compute_loss(Z_out, Y, type):
    A_out=softmax(Z_out)
    m=Y.shape[1]
    if type=='cross_entropy':
        loss=-(1/m)*np.sum(Y*np.log(A_out+1e-8))
    elif type=='mean_squared_error':
        loss=(1/m)*np.sum((Y-A_out)**2)
    else:
        raise ValueError(f"Invalid Loss function: {type}")
    return loss


def output_layer_grad(Z_L, Y, loss_type):
    A_L=softmax(Z_L)
    m = Y.shape[1]
    if loss_type=='mean_squared_error':
        dA_L=(2/m)*(A_L-Y)
        dZ_L=dA_L*A_L*(1-A_L)
    else:
        dZ_L=A_L-Y
    return dZ_L