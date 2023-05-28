import os

import torch
from torch import nn
import torch.nn.functional as F
from modules import attentions, commons
from modules.symbols import phone_set


class PhonemeAsr(nn.Module):
    """
    Model
    """

    def __init__(self, hps):
        super().__init__()
        self.hps = hps

        self.pre_net = nn.Conv1d(hps["data"]["unit_dim"], hps["model"]["prior_hidden_channels"], 1)
        self.proj_phone = nn.Conv1d(hps["model"]["prior_hidden_channels"], len(phone_set), 1)
        self.proj_tone = nn.Conv1d(hps["model"]["prior_hidden_channels"], 6, 1)

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
        pred_phoneme = self.proj_phone(x)
        pred_tone = self.proj_tone(x)
        return pred_phoneme, pred_tone

    def forward(self, phone, phone_lengths, tone, units):

        x = self.pre_net(units)
        x_mask = torch.unsqueeze(commons.sequence_mask(phone_lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        pred_phoneme = self.proj_phone(x)
        pred_tone = self.proj_tone(x)

        if phone is not None:
            loss_phoneme = F.cross_entropy(pred_phoneme, phone)
            loss_tone = F.cross_entropy(pred_tone, tone)
        else:
            loss_all = None

        return pred_phoneme, pred_tone, loss_phoneme, loss_tone
