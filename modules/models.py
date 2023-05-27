import os

import torch
from torch import nn

from modules import attentions, commons
from modules.symbols import phone_set

hps = {
  "data": {
    "unit_dim": 768,
  },
  "model": {
    "hidden_channels": 192,
    "spk_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 4,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "prior_hidden_channels": 192,
    "prior_filter_channels": 768,
    "prior_n_heads": 2,
    "prior_n_layers": 4,
    "prior_kernel_size": 3,
    "prior_p_dropout": 0.1,
    "resblock": "1",
    "use_spectral_norm": False,
    "resblock_kernel_sizes": [3,7,11],
    "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
    "upsample_rates": [8,8,4,2],
    "upsample_initial_channel": 256,
    "upsample_kernel_sizes": [16,16,8,4],
    "n_harmonic": 64,
    "n_bands": 65
  }
}

class PhonemeAsr(nn.Module):
    """
    Model
    """

    def __init__(self, hps):
        super().__init__()
        self.hps = hps

        self.pre_net = nn.Conv1d(hps["data"]["unit_dim"], hps["model"]["prior_hidden_channels"], 1)
        self.proj = nn.Conv1d(hps["model"]["prior_hidden_channels"], len(phone_set), 1)
        self.encoder = attentions.Encoder(
            hps["model"]["prior_hidden_channels"],
            hps["model"]["prior_filter_channels"],
            hps["model"]["prior_n_heads"],
            hps["model"]["prior_n_layers"],
            hps["model"]["prior_kernel_size"],
            hps["model"]["prior_p_dropout"])

    def infer(self, units):
        phone_lengths = torch.LongTensor([units.shape[2]]).to(units.device)
        x = self.pre_net(units)
        x_mask = torch.unsqueeze(commons.sequence_mask(phone_lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        x = self.proj(x)
        return x

