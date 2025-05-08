from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocities = {}
    
    def step(self):
        for layer_idx, layer in enumerate(self.model.layers):
            if layer.optimizable == True:
                for key in layer.params.keys():
                    vel_key = (layer_idx, key)
                    # 初始化动量
                    if vel_key not in self.velocities:
                        self.velocities[vel_key] = np.zeros_like(layer.params[key]) 
                    # 计算动量
                    self.velocities[vel_key] = self.mu * self.velocities[vel_key] - self.init_lr * layer.grads[key]
                    # 更新参数
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda) 
                    layer.params[key] += self.velocities[vel_key]