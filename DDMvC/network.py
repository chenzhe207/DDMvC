import torch.nn as nn
from torch.nn.functional import normalize
import torch
from torch.nn.parameter import Parameter


class Encoder(nn.Module):
    def __init__(self, input_dim, low_feature_dim):
        super(Encoder, self).__init__()
        self.encoder_layer = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, low_feature_dim),
        )

    def forward(self, x):
        return self.encoder_layer(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, low_feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(low_feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, view, input_size, low_feature_dim, high_feature_dim, class_num, dict_size):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []



        self.feature_learn = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
        )

        for v in range(view):
            self.encoders.append(Encoder(input_size[v], low_feature_dim))
            self.decoders.append(Decoder(input_size[v], low_feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.D = Parameter(torch.ones(view * high_feature_dim, class_num * dict_size))


        self.view = view
        self.class_num = class_num
        self.dict_size = dict_size

    def forward(self, xs):
        xrs = []
        fs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            f = normalize(self.feature_learn(z))
            fs.append(f)
            xrs.append(xr)
        return xrs, fs

    def calculate_s(self, xs):
        ss = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            f = normalize(self.feature_learn(z))
            f_repeat = f.repeat(1, self.view)
            s = None
            for i in range(self.class_num):
                si = torch.sum(torch.pow(torch.matmul(f_repeat, self.D[:, i * self.dict_size:(i + 1) * self.dict_size]), 2), 1, keepdim=True)
                if s is None:
                    s = si
                else:
                    s = torch.cat((s, si), 1)
            s = nn.Softmax(dim=1)(s)
            ss.append(s)
        return ss



