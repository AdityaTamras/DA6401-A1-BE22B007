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
        self.n_in=layer_dims[0]

    @property
    def init_params(self):
        params={}
        for idx, layer in enumerate(self.layers, start=0):
            params[f'W{idx}']=layer.W.copy()
            params[f'b{idx}']=layer.b.copy()
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
        input_dim=self.layers[0].W.shape[0]
        if X.ndim==1:
            X=X.reshape(-1, 1)
        row_major = (X.shape[1] == input_dim)
        col_major = (X.shape[0] == input_dim)
        if row_major and not col_major:
            X_col=X.T
        else:
            X_col=X
        cache={'A_0': X_col}
        L=self.num_layers
        A_prev=X_col

        for idx, layer in enumerate(self.layers, start=1):
            A=layer.forward(A_prev)
            cache[f'A_{idx}']=A
            A_prev=A

        self.hidden_activations=[cache[f'A_{l}'] for l in range(1, L)]
        self.cache=cache
        if row_major and not col_major:
            return cache[f'A_{L}'].T
        return cache[f'A_{L}']
    
    def compute_loss(self,Z_out, Y, type='cross_entropy'):
        return compute_loss(Z_out, Y, type)

    def backward(self,Z_L, y, loss_type='cross_entropy'):
        cache=self.cache
        grads={}
        n_out=self.layers[-1].W.shape[1]
        if Z_L.ndim == 1:
            Z_L = Z_L.reshape(-1, 1)
        if Z_L.shape[0] != n_out and Z_L.shape[1] == n_out:
            Z_L = Z_L.T

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.shape[0] != n_out and y.shape[1] == n_out:
            y = y.T

        m=y.shape[1] if y.ndim>1 else y.shape[0]
        L=self.num_layers

        grad_W_list = []
        grad_b_list = []
        dZ_L=output_layer_grad(Z_L, y, loss_type)
        output_layer=self.layers[L-1]
        dA_prev=output_layer.backward(dZ_L, m)
        grad_W_list.append(self.layers[L-1].grad_W)
        grad_b_list.append(self.layers[L-1].grad_b)

        for l in reversed(range(1, L)):
            dA_prev=self.layers[l - 1].backward(dA_prev, m)
            grad_W_list.append(self.layers[l-1].grad_W)
            grad_b_list.append(self.layers[l-1].grad_b)

        self.grad_W=np.empty(len(grad_W_list), dtype=object)
        self.grad_b=np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(reversed(grad_W_list), reversed(grad_b_list))):
            self.grad_W[i]=gw
            self.grad_b[i]=gb

        self.layer_grad_norms = [np.linalg.norm(self.grad_W[i]) for i in range(L)]
        return self.grad_W, self.grad_b