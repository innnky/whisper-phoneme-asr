import torch

from config import hps
from modules.models import PhonemeAsr
from utils import load_checkpoint

checkpoint_path = ''
recognition_model = PhonemeAsr(hps)
recognition_model.eval()
load_checkpoint(checkpoint_path, recognition_model)
torch.save(recognition_model.cpu().state_dict(), "assets/recognition_model.pth")

