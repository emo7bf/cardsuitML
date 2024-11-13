import torch.nn as nn
import torch.nn.functional as F


class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()
        # (input channels (ie, 3 colors), # output channels , filter size nxn )
        self.conv1 = nn.Conv2d(3, 32, 10)
        # 3*32
        self.conv2 = nn.Conv2d(32, 128, 10)
        # self.conv3 = nn.Conv2d(64, 128, 3)

        self.fc1 = nn.Linear(128, 2)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x1 = self.pool(F.relu(self.conv1(x)))

        x2 = self.pool(F.relu(self.conv2(x1)))

        # x3 = self.pool(F.relu(self.conv3(x2)))

        bs, _, _, _ = x2.shape
        x = F.adaptive_avg_pool2d(x2, 1).reshape(bs, -1)
        x = self.fc1(x)
        return x

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook