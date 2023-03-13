from abc import ABC, abstractmethod

import numpy as np
import scipy.special as ssp
from tqdm import tqdm


################################################################
#                         Base Classe                          #
################################################################

class Module(ABC):
    """
    Defines NN Module with tunable params
    """

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, grad):
        pass

    @abstractmethod
    def zero_grad(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Activation(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, grad):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Loss(ABC):
    @abstractmethod
    def forward(self, x, target):
        pass

    @abstractmethod
    def backward(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Optimizer(ABC):
    @abstractmethod
    def step(self, loss):
        pass


################################################################
#                          Class                               #
################################################################

class Linear(Module):
    def __init__(self, in_features, out_features, use_bias=True):
        self.grad_x = None
        self.grad_bias = None
        self.grad_weight = None
        self.x = None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.weight = np.random.normal(0.0, pow(self.in_features, -0.5), (self.out_features, self.in_features))
        if use_bias:
            self.bias = np.random.normal(0.0, pow(self.in_features, -0.5), (self.out_features,))
        else:
            self.bias = None

    def forward(self, x):
        self.x = x.copy()
        if self.use_bias:
            return np.dot(x, self.weight.T) + self.bias
        else:
            return np.dot(x, self.weight.T)

    def backward(self, grad):
        """
        :param grad: delta
        :return: delta*weight
        delta, which is partial derivative of loss w.r.t. z
        z = w * x + b
        a = f(z)
        loss = C(a)
        """
        # self.grad_weight = np.dot(self.x.T, grad)  # x * delta or a_{l-1} * delta
        self.grad_weight = np.zeros_like(self.weight)
        for i in range(grad.shape[0]):  # for all batches
            d = grad[i].reshape(-1, 1)
            x = self.x[i].reshape(1, -1)
            self.grad_weight += np.dot(d, x)
        self.grad_weight /= grad.shape[0]  # average over batch size

        if self.use_bias:
            self.grad_bias = np.sum(grad, axis=0, keepdims=True)  # sum over batch size
        self.grad_x = np.dot(grad, self.weight)  # delta * w, so that d_{l-1} = f'(z_{l-1}) * (d_{l} * w_{l})
        return self.grad_x

    def zero_grad(self):
        self.grad_x = None
        self.grad_bias = None
        self.grad_weight = None
        self.x = None


class LeakyReLU(Activation):
    def __init__(self):
        self.grad_x = None
        self.x = None

    @staticmethod
    def _leaky_relu(x):
        return np.maximum(0.01 * x, x)

    def forward(self, x):
        self.x = x.copy()
        return self._leaky_relu(x)

    def backward(self, grad):
        # clip gradient to avoid exploding gradient
        grad = np.clip(grad, -1, 1)
        self.grad_x = grad * (self.x > 0) + 0.01 * grad * (self.x <= 0)
        return self.grad_x


class ReLU(Activation):
    def __init__(self):
        self.grad_x = None
        self.x = None

    def forward(self, x):
        self.x = x.copy()
        return np.maximum(0, x)

    def backward(self, grad):
        # clip gradient to avoid exploding gradient
        grad = np.clip(grad, -1, 1)
        self.grad_x = grad * (self.x > 0)
        return self.grad_x


class Sigmoid(Activation):
    def __init__(self):
        self.grad_x = None
        self.x = None

    def forward(self, x):
        self.x = x.copy()
        return ssp.expit(x)

    def backward(self, grad):
        fp = ssp.expit(self.x)
        self.grad_x = grad * fp * (1 - fp)
        return self.grad_x


class MLP(Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, activation=LeakyReLU, use_bias=True):
        self.layers = [Linear(input_size, hidden_size, use_bias), activation()]
        for _ in range(num_layers - 2):
            self.layers.append(Linear(hidden_size, hidden_size, use_bias))
            self.layers.append(activation())
        self.layers.append(Linear(hidden_size, output_size, use_bias))
        self.layers.append(activation())
        # self.layers.append(Sigmoid())  # todo check

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def zero_grad(self):
        for layer in self.layers:
            if isinstance(layer, Module):
                layer.zero_grad()


class MSE(Loss):
    def __init__(self):
        self.grad_x = None
        self.x = None
        self.target = None

    def forward(self, x, target):
        self.x = x.copy()
        self.target = target.copy()
        return 0.5 * np.sum(np.square(x - target))

    def backward(self):
        self.grad_x = self.x - self.target
        return self.grad_x


class SGD(Optimizer):
    def __init__(self, network, lr, momentum=0):
        self.network = network  # reference to network
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(layer.weight) if isinstance(layer, Linear) else None for layer in
                           self.network.layers]

    def step(self, loss_grad):
        self.network.backward(loss_grad)  # update grad

        for i, layer in enumerate(self.network.layers):
            if isinstance(layer, Module):
                if hasattr(layer, "weight"):
                    self.velocities[i] = self.momentum * self.velocities[i] + layer.grad_weight
                    layer.weight -= self.lr * self.velocities[i]
                    if layer.bias is not None:
                        layer.bias -= (self.lr * layer.grad_bias).squeeze()

        self.network.zero_grad()  # clear gradient


class Dataloader:
    def __init__(self, file_path, batch_size=1):
        self.file_path = file_path
        self.batch_size = batch_size
        with open(self.file_path, "r") as f:
            self.data_list = f.readlines()
        self.num_samples = len(self.data_list)
        self.num_batches = self.num_samples // self.batch_size

        # process data
        self.labels = np.zeros((self.num_samples,), dtype=int)
        self.images = np.zeros((self.num_samples, 28 * 28))
        for i, line in enumerate(self.data_list):
            line = line.split(",")
            self.labels[i] = int(line[0])
            self.images[i] = (np.asfarray(line[1:]) / 255.0 * 0.99) + 0.01

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for i in range(self.num_batches):
            images = self.images[i * self.batch_size:(i + 1) * self.batch_size]
            labels = self.labels[i * self.batch_size:(i + 1) * self.batch_size]
            yield images, labels


################################################################
#                         Util Funcs                           #
################################################################


def test(model, test_loader):
    correct = 0

    for inputs, target in tqdm(test_loader):
        # inputs = inputs.reshape(-1, 28 * 28)
        # inputs = inputs.numpy()  # todo check
        output = model(inputs)
        pred = np.argmax(output)
        correct += pred == target

    return correct / len(test_loader)


def get_activation(activation_name):
    if activation_name == "LeakyReLU":
        return LeakyReLU
    elif activation_name == "ReLU":
        return ReLU
    elif activation_name == "sigmoid":
        return Sigmoid
    else:
        raise NotImplementedError(f"Activation {activation_name} is not implemented")


def one_hot(label, num_classes=10):
    return np.eye(num_classes)[label]


################################################################
#                         Instances                            #
################################################################

test_loader = Dataloader(file_path='./mnist_test.csv', batch_size=1)
