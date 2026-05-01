import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, L: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 4, L)
            out = self.net(dummy)
            self.flat = out.shape[1] * out.shape[2]
        self.fc = nn.Sequential(
            nn.Linear(self.flat, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        z = self.net(x)
        z = z.reshape(z.size(0), -1)
        return self.fc(z).squeeze(1)


class DeepSTARR(nn.Module):
    """DeepSTARR model from de Almeida et al., 2022.
    https://www.nature.com/articles/s41588-022-01048-5
    """

    def __init__(self, output_dim, d=256,
                 conv1_filters=None, learn_conv1_filters=True,
                 conv2_filters=None, learn_conv2_filters=True,
                 conv3_filters=None, learn_conv3_filters=True,
                 conv4_filters=None, learn_conv4_filters=True):
        super().__init__()
        assert not (conv1_filters is None and not learn_conv1_filters)
        assert not (conv2_filters is None and not learn_conv2_filters)
        assert not (conv3_filters is None and not learn_conv3_filters)
        assert not (conv4_filters is None and not learn_conv4_filters)

        self.activation = nn.ReLU()
        self.dropout4 = nn.Dropout(0.4)
        self.flatten = nn.Flatten()

        self.init_conv1_filters = conv1_filters
        self.init_conv2_filters = conv2_filters
        self.init_conv3_filters = conv3_filters
        self.init_conv4_filters = conv4_filters

        def _make_conv_param(filters, default_shape, learn):
            if filters is not None:
                if learn:
                    return nn.Parameter(torch.Tensor(filters))
                else:
                    return filters  # registered as buffer by caller
            p = nn.Parameter(torch.zeros(*default_shape))
            nn.init.kaiming_normal_(p)
            return p

        # Layer 1
        if conv1_filters is not None:
            if learn_conv1_filters:
                self.conv1_filters = nn.Parameter(torch.Tensor(conv1_filters))
            else:
                self.register_buffer("conv1_filters", torch.Tensor(conv1_filters))
        else:
            self.conv1_filters = nn.Parameter(torch.zeros(d, 4, 7))
            nn.init.kaiming_normal_(self.conv1_filters)
        self.batchnorm1 = nn.BatchNorm1d(d)
        self.activation1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(2)

        # Layer 2
        if conv2_filters is not None:
            if learn_conv2_filters:
                self.conv2_filters = nn.Parameter(torch.Tensor(conv2_filters))
            else:
                self.register_buffer("conv2_filters", torch.Tensor(conv2_filters))
        else:
            self.conv2_filters = nn.Parameter(torch.zeros(60, d, 3))
            nn.init.kaiming_normal_(self.conv2_filters)
        self.batchnorm2 = nn.BatchNorm1d(60)
        self.activation2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(2)

        # Layer 3
        if conv3_filters is not None:
            if learn_conv3_filters:
                self.conv3_filters = nn.Parameter(torch.Tensor(conv3_filters))
            else:
                self.register_buffer("conv3_filters", torch.Tensor(conv3_filters))
        else:
            self.conv3_filters = nn.Parameter(torch.zeros(60, 60, 5))
            nn.init.kaiming_normal_(self.conv3_filters)
        self.batchnorm3 = nn.BatchNorm1d(60)
        self.activation3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(2)

        # Layer 4
        if conv4_filters is not None:
            if learn_conv4_filters:
                self.conv4_filters = nn.Parameter(torch.Tensor(conv4_filters))
            else:
                self.register_buffer("conv4_filters", torch.Tensor(conv4_filters))
        else:
            self.conv4_filters = nn.Parameter(torch.zeros(120, 60, 3))
            nn.init.kaiming_normal_(self.conv4_filters)
        self.batchnorm4 = nn.BatchNorm1d(120)
        self.activation4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool1d(2)

        # Fully connected
        self.fc5 = nn.LazyLinear(256, bias=True)
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.activation5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.4)

        self.fc6 = nn.Linear(256, 256, bias=True)
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.activation6 = nn.ReLU()
        self.dropout6 = nn.Dropout(0.4)

        self.fc7 = nn.Linear(256, output_dim)

    def get_which_conv_layers_transferred(self):
        layers = []
        for i, f in enumerate([self.init_conv1_filters, self.init_conv2_filters,
                                self.init_conv3_filters, self.init_conv4_filters], 1):
            if f is not None:
                layers.append(i)
        return layers

    def forward(self, x):
        cnn = torch.conv1d(x, self.conv1_filters, stride=1, padding="same")
        cnn = self.activation1(self.batchnorm1(cnn))
        cnn = self.maxpool1(cnn)

        cnn = torch.conv1d(cnn, self.conv2_filters, stride=1, padding="same")
        cnn = self.activation2(self.batchnorm2(cnn))
        cnn = self.maxpool2(cnn)

        cnn = torch.conv1d(cnn, self.conv3_filters, stride=1, padding="same")
        cnn = self.activation3(self.batchnorm3(cnn))
        cnn = self.maxpool3(cnn)

        cnn = torch.conv1d(cnn, self.conv4_filters, stride=1, padding="same")
        cnn = self.activation4(self.batchnorm4(cnn))
        cnn = self.maxpool4(cnn)

        cnn = self.dropout5(self.activation5(self.batchnorm5(self.fc5(self.flatten(cnn)))))
        cnn = self.dropout5(self.activation6(self.batchnorm6(self.fc6(cnn))))
        return self.fc7(cnn).squeeze(1)
