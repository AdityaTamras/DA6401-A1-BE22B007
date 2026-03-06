import numpy as np

class Optimizer:
    def __init__(self, method, lr, wd):
        self.method = method
        self.lr=lr
        self.wd=wd
        self.v={}
        self.m={}

    def update_parameters(self, params, grads):
        for key in params.keys():
            if key not in self.v:
                self.v[key]=np.zeros_like(params[key])
                self.m[key]=np.zeros_like(params[key])

            dw=grads[f'd{key}']
            if 'w' in key:
                dw=dw+(self.wd*params[key])

            if self.method=='sgd':
                params[key]-=self.lr*dw
            elif self.method=='momentum':
                gamma = 0.9
                self.v[key]=(gamma*self.v[key])+(self.lr*dw)
                params[key]-=self.v[key]
            elif self.method=='nag':
                mu=0.9
                v_prev=self.v[key].copy()
                self.v[key]=(mu*self.v[key])-(self.lr*dw)
                params[key]+=(-mu*v_prev)+((1+mu)*self.v[key])
            elif self.method == 'rmsprop':
                beta, eps = 0.9, 1e-8
                self.v[key]=(beta*self.v[key])+((1-beta)*(dw**2))
                params[key]-=(self.lr/(np.sqrt(self.v[key])+eps))*dw
        return params