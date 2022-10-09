import torch
import torchaudio
import omegaconf


class SpecScaler(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x.clamp_(1e-9, 1e9))


class FeatureExtractor(torch.nn.Module):
    def __init__(self, conf: omegaconf.DictConfig):

        super().__init__()

        self.model = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=conf.sample_rate, **conf.features
            ),
            SpecScaler(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Model(torch.nn.Module):
    def __init__(self, conf: omegaconf.DictConfig):

        super().__init__()

        activation = getattr(torch.nn, conf.model.activation)()

        features = conf.features.n_mels
        n_classes = len(conf.idx_to_keyword)

        module_list = []

        for kernel_size, stride, channels in zip(
            conf.model.kernels, conf.model.strides, conf.model.channels
        ):

            module_list.extend(
                [
                    torch.nn.Conv1d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=kernel_size,
                        stride=stride,
                        groups=features,
                    ),
                    activation,
                    torch.nn.Conv1d(
                        in_channels=features, out_channels=channels, kernel_size=1
                    ),
                    torch.nn.BatchNorm1d(num_features=channels),
                    activation,
                    torch.nn.MaxPool1d(kernel_size=stride),
                ]
            )

            features = channels

        module_list.extend(
            [
                torch.nn.AdaptiveAvgPool1d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(channels, conf.model.hidden_size),
                activation,
                torch.nn.Linear(conf.model.hidden_size, n_classes),
                torch.nn.LogSoftmax(-1),
            ]
        )

        self.model = torch.nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
