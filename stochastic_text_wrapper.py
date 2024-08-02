import torch
from torch import nn
from config.base_config import Config
from modules.stochastic_module import StochasticText


class StochasticTextWrapper(nn.Module):
    def __init__(self, config: Config):
        super(StochasticTextWrapper, self).__init__()
        self.stochastic_text = StochasticText(config)

    def forward(self, text_features, video_features):
        # Check if video_features has 3 dimensions, if not adjust it
        if video_features.dim() == 2:
            video_features = video_features.unsqueeze(1).repeat(1, self.stochastic_text.config.num_frames, 1)

        return self.stochastic_text(text_features, video_features)
