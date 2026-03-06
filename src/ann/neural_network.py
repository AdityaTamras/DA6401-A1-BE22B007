import numpy as np
from .neural_layer import Layer
from .objective_functions import compute_loss, output_layer_grad
from .activations import activation_func

class NeuralNetwork:
    def __init__(self, layer_dims, weight_init='xavier', activation_function='relu'):
        if hasattr(layer_dims,' __dict__') and not isinstance(layer_dims, (list, tuple)):
            args=layer_dims
            weight_init=getattr(args, 'weight_init',   weight_init)
            activation_function=getattr(args, 'activation',    activation_function)
            hidden_sizes=list(map(int, args.hidden_size)) if isinstance(args.hidden_size, (list, tuple)) else list(map(int, args.hidden_size.split()))
            layer_dims = [784] + hidden_sizes + [10]
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
        for idx, layer in enumerate(self.layers, start=1):
            params[f'w_{idx}']=layer.W
            params[f'b_{idx}']=layer.b
        return params

    def get_weights(self):
        return {key: val.copy() for key, val in self.init_params.items()}
    
    def set_weights(self, weights):
        for key in weights:
            self.init_params[key]=weights[key]

    def forward(self, X):
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

    def backward(self, Z_L, y, loss_type, cache):
        grads={}
        m=y.shape[1]
        L=self.num_layers

        dZ_L=output_layer_grad(Z_L, y, loss_type)
        output_layer=self.layers[L-1]
        dA_prev=output_layer.backward(dZ_L, m)
        grads[f'dw_{L}']=output_layer.grad_W
        grads[f'db_{L}']=output_layer.grad_b

        for l in reversed(range(1, L)):
            layer=self.layers[l-1]
            dA_prev=layer.backward(dA_prev, m)
            grads[f'dw_{l}']=layer.grad_W
            grads[f'db_{l}']=layer.grad_b

        self.layer_grad_norms = [np.linalg.norm(grads[f'dw_{l}']) for l in range(1, L+1)]
        return grads