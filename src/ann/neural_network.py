import argparse
import numpy as np
from .neural_layer import Layer
from .objective_functions import compute_loss, output_layer_grad
from .activations import activation_func

class NeuralNetwork:
    def __init__(self, layer_dims, weight_init='xavier', activation_function='relu'):
        if isinstance(layer_dims, argparse.Namespace):
            args=layer_dims
            weight_init=getattr(args, 'weight_init',   weight_init)
            activation_function=getattr(args, 'activation',    activation_function)
            hidden_size=getattr(args, 'hidden_size', None)
            input_dim=getattr(args, 'input_dim', 784)
            output_dim=getattr(args, 'output_dim', 10)

            if hidden_size is None:
                hidden_sizes=[128]
            elif isinstance(hidden_size, (list, tuple)):
                hidden_sizes = [int(x) for x in hidden_size]
            else:
                hidden_sizes = [int(x) for x in str(hidden_size).split()]

            layer_dims = [int(input_dim)] + hidden_sizes + [int(output_dim)]

        layer_dims = [int(x) for x in layer_dims]

        self.num_layers=len(layer_dims)-1
        self.activation_function=activation_function
        self.hidden_activations=[]
        self.layer_grad_norms=[]

        self.layers=[]
        for i in range(1, len(layer_dims)):
            activation='linear' if i==len(layer_dims)-1 else activation_function
            self.layers.append(
                Layer(n_in=layer_dims[i-1], n_out=layer_dims[i], activation=activation, w_init=weight_init)
            )

    @property
    def init_params(self):
        params={}
        for idx, layer in enumerate(self.layers, start=0):
            params[f'W{idx}']=layer.W
            params[f'b{idx}']=layer.b
        return params

    def get_weights(self):
        d={}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d
    
    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key=f"W{i}"
            b_key=f"b{i}"
            if w_key in weight_dict:
                layer.W=weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b=weight_dict[b_key].copy()

    def forward(self, X):
        input_dim=self.layers[0].W.shape[1]
        X=np.array(X)
        if X.ndim==1:
            X=X.reshape(input_dim, -1)
        elif X.ndim==2:
            if X.shape[0]==input_dim:
                pass
            elif X.shape[1] == input_dim:
                X = X.T
        
        cache={}
        L=self.num_layers
        A_prev=X
        cache['A_0']=X

        for idx, layer in enumerate(self.layers, start=1):
            A=layer.forward(A_prev)
            cache[f'A_{idx}']=A
            A_prev=A

        self.hidden_activations=[cache[f'A_{l}'] for l in range(1, L)]
        return cache[f'A_{L}'], cache
    
    def compute_loss(self, Z_out, Y, type):
        return compute_loss(Z_out, Y, type)

    def backward(self, Z_L, y, loss_type='cross_entropy', cache=None):
        grads={}
        m=y.shape[1] if y.ndim > 1 else y.shape[0]
        L=self.num_layers

        dZ_L=output_layer_grad(Z_L, y, loss_type)
        output_layer=self.layers[L-1]
        dA_prev=output_layer.backward(dZ_L, m)
        grads[f'dW{L-1}']=output_layer.grad_W
        grads[f'db{L-1}']=output_layer.grad_b

        for l in reversed(range(1, L)):
            layer=self.layers[l-1]
            dA_prev=layer.backward(dA_prev, m)
            grads[f'dW{l-1}']=layer.grad_W
            grads[f'db{l-1}']=layer.grad_b

        self.layer_grad_norms = [np.linalg.norm(grads[f'dW{l}']) for l in range(0, L)]
        return grads