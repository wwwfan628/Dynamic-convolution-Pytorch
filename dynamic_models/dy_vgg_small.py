import torch.nn as nn
from dynamic_conv import Dynamic_conv2d


class dy_vgg_small(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, input_channel=3, n_classes=1000):
        super(dy_vgg_small, self).__init__()
        self.features = nn.Sequential(
            Dynamic_conv2d(input_channel, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            Dynamic_conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Dynamic_conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            Dynamic_conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Dynamic_conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            Dynamic_conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x