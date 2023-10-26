import torch
from torch import nn

class VGG16(nn.Module):
    """This class implements the VGG-16 architecture in PyTorch"""

    def __init__(self, activation_str="relu", initialization="xavier_uniform"):
        """
            Constructor for the VGG16 class.

            activation_str: string, default "relu"
            Activation function to use.
        """
        super(VGG16, self).__init__()

        self.n_classes = 10 # WRITE CODE HERE
        self.activation_str = activation_str
        self.initialization = initialization

        self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding="same") # WRITE CODE HERE
        self.conv_layer_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same") # WRITE CODE HERE
        self.conv_layer_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same") # WRITE CODE HERE
        self.conv_layer_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same") # WRITE CODE HERE
        self.conv_layer_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding="same") # WRITE CODE HERE
        self.conv_layer_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding="same") # WRITE CODE HERE
        self.conv_layer_7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding="same") # WRITE CODE HERE
        self.conv_layer_8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding="same") # WRITE CODE HERE
        self.conv_layer_9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same") # WRITE CODE HERE
        self.conv_layer_10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same") # WRITE CODE HERE
        self.conv_layer_11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same") # WRITE CODE HERE
        self.conv_layer_12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same") # WRITE CODE HERE
        self.conv_layer_13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same") # WRITE CODE HERE

        # Add 2D batch normalization after every convolutional layer
        self.conv_layer_1_bn = nn.BatchNorm2d(num_features=64) # WRITE CODE HERE
        self.conv_layer_2_bn = nn.BatchNorm2d(num_features=64) # WRITE CODE HERE
        self.conv_layer_3_bn = nn.BatchNorm2d(num_features=128) # WRITE CODE HERE
        self.conv_layer_4_bn = nn.BatchNorm2d(num_features=128) # WRITE CODE HERE
        self.conv_layer_5_bn = nn.BatchNorm2d(num_features=256) # WRITE CODE HERE
        self.conv_layer_6_bn = nn.BatchNorm2d(num_features=256) # WRITE CODE HERE
        self.conv_layer_7_bn = nn.BatchNorm2d(num_features=256) # WRITE CODE HERE
        self.conv_layer_8_bn = nn.BatchNorm2d(num_features=512) # WRITE CODE HERE
        self.conv_layer_9_bn = nn.BatchNorm2d(num_features=512) # WRITE CODE HERE
        self.conv_layer_10_bn = nn.BatchNorm2d(num_features=512) # WRITE CODE HERE
        self.conv_layer_11_bn = nn.BatchNorm2d(num_features=512) # WRITE CODE HERE
        self.conv_layer_12_bn = nn.BatchNorm2d(num_features=512) # WRITE CODE HERE
        self.conv_layer_13_bn = nn.BatchNorm2d(num_features=512) # WRITE CODE HERE

        self.max_pool_layer_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # WRITE CODE HERE
        self.max_pool_layer_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # WRITE CODE HERE
        self.max_pool_layer_3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # WRITE CODE HERE
        self.max_pool_layer_4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # WRITE CODE HERE
        self.max_pool_layer_5 = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # WRITE CODE HERE

        self.fc_1 = nn.Linear(in_features=512, out_features=256, bias=True) # WRITE CODE HERE
        self.fc_2 = nn.Linear(in_features=256, out_features=128, bias=True) # WRITE CODE HERE
        self.fc_3 = nn.Linear(in_features=128, out_features=self.n_classes, bias=True) # WRITE CODE HERE

        # Initialize the weights of each trainable layer of your network using xavier_uniform initialization
        # WRITE CODE HERE
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if self.initialization == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

    def activation(self, input):
        """
            input: Tensor
            Input on which the activation is applied.

            Output: Result of activation function applied on input.
            E.g. if self.activation_str is "relu", return relu(input).
        """
        if self.activation_str == "relu":
            # WRITE CODE HERE
            return nn.functional.relu(input)
        elif self.activation_str == "tanh":
            # WRITE CODE HERE
            return nn.functional.tanh(input)
        else:
            raise Exception("Invalid activation")
        return 0

    def forward(self, x):
        """
            x: Tensor
            Input to the network.

            Outputs: Returns the output of the forward pass of the network.
        """
        # WRITE CODE HERE
        x = self.conv_layer_1(x)
        x = self.conv_layer_1_bn(x)
        x = self.activation(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_2_bn(x)
        x = self.activation(x)
        x = self.max_pool_layer_1(x)

        x = self.conv_layer_3(x)
        x = self.conv_layer_3_bn(x)
        x = self.activation(x)
        x = self.conv_layer_4(x)
        x = self.conv_layer_4_bn(x)
        x = self.activation(x)
        x = self.max_pool_layer_2(x)

        x = self.conv_layer_5(x)
        x = self.conv_layer_5_bn(x)
        x = self.activation(x)
        x = self.conv_layer_6(x)
        x = self.conv_layer_6_bn(x)
        x = self.activation(x)
        x = self.conv_layer_7(x)
        x = self.conv_layer_7_bn(x)
        x = self.activation(x)
        x = self.max_pool_layer_3(x)

        x = self.conv_layer_8(x)
        x = self.conv_layer_8_bn(x)
        x = self.activation(x)
        x = self.conv_layer_9(x)
        x = self.conv_layer_9_bn(x)
        x = self.activation(x)
        x = self.conv_layer_10(x)
        x = self.conv_layer_10_bn(x)
        x = self.activation(x)
        x = self.max_pool_layer_4(x)

        x = self.conv_layer_11(x)
        x = self.conv_layer_11_bn(x)
        x = self.activation(x)
        x = self.conv_layer_12(x)
        x = self.conv_layer_12_bn(x)
        x = self.activation(x)
        x = self.conv_layer_13(x)
        x = self.conv_layer_13_bn(x)
        x = self.activation(x)
        x = self.max_pool_layer_5(x)

        x = x.view(x.size(0), -1)

        x = self.fc_1(x)
        x = self.activation(x)
        x = self.fc_2(x)
        x = self.activation(x)
        x = self.fc_3(x)
        x = nn.functional.softmax(x, dim=1)

        return x
