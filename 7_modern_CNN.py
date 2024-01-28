import torch
from torch import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    #通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean)/torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X-mean)**2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X-mean)**2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean)/torch.sqrt(var + eps)
        moving_mean = momentum*moving_mean + (1.0 - momentum)*mean
        moving_var = momentum*moving_var + (1.0 - momentum)*var
    Y = gamma * X_hat + beta #缩放和移位
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims=4, affine=True):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var,\
                                                          eps=1e-5, momentum=0.9)
        return Y

def main():
    input = torch.randn(20, 100, 35, 45)
    input_mean = input.mean(dim=(0, 2, 3), keepdim=True)
    input_var = ((input-input_mean)**2).mean(dim=(0, 2, 3))
    print(input_var.shape)
    print(input_var)
    # output = batch_norm(input)
    # m = nn.BatchNorm2d(100)
    m = BatchNorm(num_features=100)
    output = m(input)
    # print(m.gamma.reshape((-1,)), m.beta.reshape((-1,)))
    output_mean = output.mean(dim=(0, 2, 3), keepdim=True)
    output_var = ((output-output_mean)**2).mean(dim=(0, 2, 3))
    print(output_var)
    print(output.shape)

if __name__ == '__main__':
    main()