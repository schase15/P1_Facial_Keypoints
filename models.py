# Build Neural Network model

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional layers - input size is (1, 224, 224)
        self.conv1 = nn.Conv2d(1, 32, 3)        # Output size is (224-3)/1 +1 = 222 -> (32, 222, 222) -> After max pool (32, 111, 111)
        self.conv2 = nn.Conv2d(32, 64, 3)       # Output size is (111-3)/1 +1 = 109 -> (64, 109, 109) -> After max pool (64, 54, 54)
        self.conv3 = nn.Conv2d(64, 128, 3)      # Output size is (54-3)/1 +1 = 52 -> (128, 52, 52) -> After max pool, (128, 26, 26)
        self.conv4 = nn.Conv2d(128, 256, 3)     # Output size is (26-3)/1 +1 = 24 -> (256, 24, 24) -> After max pool, (256, 12, 12)

        # Max Pool layer
        self.pool = nn.MaxPool2d(2,2)

        # Linear layers  - fully connected
        self.fc1 = nn.Linear(256*12*12, 1000)      # 256 outputs * 12*12 map size
        self.fc2 = nn.Linear(1000, 136)            # Output is 136, 2 values for each of the 668 keypoints (x, y) pairs

        # Dropout layer - to go between each conv layer
        self.fc_drop = nn.Dropout(p=0.4)

        # Batch normalization - to use between fully connected layers
        self.bn = nn.BatchNorm1d(1000)

        
    def forward(self, x):

        # Define Forward function of the NN

        # Convolutional layers with MaxPooling and dropout
        x = self.fc_drop(self.pool(F.relu(self.conv1(x))))
        x = self.fc_drop(self.pool(F.relu(self.conv2(x))))
        x = self.fc_drop(self.pool(F.relu(self.conv3(x))))
        x = self.fc_drop(self.pool(F.relu(self.conv4(x))))

        # Prep for linear layers
        x = x.view(x.size(0), -1)

        # Linear layer with dropout
        x = self.fc_drop(F.relu(self.bn(self.fc1(x))))

        # Output dense layer
        x = self.fc2(x)
        
        # Return x which has gone through the model
        return x
