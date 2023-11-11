import torch
from torch import nn
import torchvision.transforms as transforms
__all__ = ['lstm']
torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Lstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=10):
        super(Lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size*3, hidden_size, num_layers, batch_first=True)
        self.line=nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 30, 'lr': 1e-2},
            {'epoch': 60, 'lr': 1e-3, 'weight_decay': 0},
            {'epoch': 90, 'lr': 1e-4}
        ]


    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # 初始化hidden和memory cell参数
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward propagate lstm
        out, (h_n, h_c) = self.lstm(x, (h0, c0))

        # 选取最后一个时刻的输出
        out = self.line(out[:, -1, :])
        out = self.fc(out)
        return out

    @staticmethod
    def regularization(model, weight_decay=1e-4):
        l2_params = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                l2_params += m.weight.pow(2).sum()
                if m.bias is not None:
                    l2_params += m.bias.pow(2).sum()
        return weight_decay * 0.5 * l2_params




def lstm(**kwargs):
    num_classes, dataset = map(
    kwargs.get, ['num_classes', 'dataset'])
    input_size = 32  # rnn 每步输入值 / 图片每行像素
    hidden_size = 128
    num_layers = 3
    sequence_length = 28

    num_classes = num_classes or 10
    return Lstm(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)