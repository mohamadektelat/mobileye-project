try:
    from Configurations import *
except ImportError:
    print("Need to fix the installation")
    raise


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.first_conv_layer = nn.Conv2d(3, 6, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.second_conv_layer = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.first_full_connected_layer = nn.Linear(16 * 2448, 120)
        self.second_full_connected_layer = nn.Linear(120, 64)
        self.third_full_connected_layer = nn.Linear(64, 1)

    def forward(self, x):
        first_conv_result = self.pool(f.relu(self.first_conv_layer(x)))
        second_conv_result = self.pool(f.relu(self.second_conv_layer(first_conv_result)))
        flatten_result = second_conv_result.view(-1, 16 * 2448)
        first_linear_layer_result = f.relu(self.first_full_connected_layer(flatten_result))
        second_linear_layer_result = f.relu(self.second_full_connected_layer(first_linear_layer_result))
        third_linear_layer_result = self.third_full_connected_layer(second_linear_layer_result)
        return third_linear_layer_result
