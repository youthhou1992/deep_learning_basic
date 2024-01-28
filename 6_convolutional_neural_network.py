import torch
from torch import nn
# from d2l import torch as d2l

def corr2d(X, K):
    print("x shape: ", X.shape)
    print("k shape: ", K.shape)
    h, w = K.shape
    Y = torch.zeros((X.shape[0]-h+1, X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            window = X[i:i+h, j:j+w]
            Y[i][j] = (window*K).sum()
    return Y

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
    
def main():
    X = torch.arange(16)
    X = X.reshape((4, 4))
    K = torch.Tensor([[1, 2],[1,2]])
    Y = corr2d(X, K)
    print(Y)

def main2():
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    K = torch.Tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    print(Y.shape)
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    lr = 3e-2
    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat - Y)**2
        conv2d.zero_grad()
        # print(l)
        l.sum().backward() #l是四维张量，求梯度需要求和
        conv2d.weight.data[:] -= lr * conv2d.weight.grad 
        if (i+1) %2 == 0:
            print("epoch {}, loss {}".format(i+1, l.sum()))
        break
    print(conv2d.weight.data.shape)

if __name__ == '__main__':
    # main()
    main2()