import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, normalization_method="BN"):
        """
        Default normalization = batch normalization
        """
        super(Net, self).__init__()
        #Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3,padding=1), #Input (28, 28, 1) > Output (28, 28, 8)
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.03)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1), #Input (28, 28, 8) > Output (28, 28, 16)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.03)
        )

        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2,2),  #Input (28, 28, 8) > Output (14, 14, 16)

            nn.Conv2d(16, 16, 3, padding=1),  #Input (14, 14, 8) > Output (14, 14, 16)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.03),

            nn.Conv2d(16, 8, 3, padding=1), #Input (14, 14, 16) > Output (14, 14, 8)
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.03)
        )

        self.trans2 = nn.Sequential(
            nn.MaxPool2d(2,2), #Input (14, 14, 8) > Output (7, 7, 8)

            nn.Conv2d(8, 16, 3), #Input (7, 7, 8) > Output (5, 5, 16)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.03),

            nn.Conv2d(16, 16, 3), #Input (5, 5, 16) > Output (3, 3, 16)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.03),

            nn.Conv2d(16, 16, 1), #Input (3, 3, 16) > Output (3, 3, 10)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.03),

            nn.Conv2d(16, 10, 1), #Input (3, 3, 16) > Output (3, 3, 10)
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.03),

            nn.AvgPool2d(3) #Input (3, 3, 10) > Output (1, 1, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.trans1(x)
        x = self.trans2(x)
        x = x.view(-1, 10)
        return F.log_softmax(x,dim=-1)

def normalizer(method, out_channels):
    if method not in ["BN", "GN", "LN"]:
        raise ValueError("Invalid method of normalization")

    if method == "BN":
        return nn.BatchNorm2d(out_channels)
    elif method == "LN":
        return nn.GroupNorm(1, out_channels)
    else:
        return nn.GroupNorm(5, out_channels)