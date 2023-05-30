import torch
from .whisper_encoder import AudioEncoder,  log_mel_spectrogram, pad_or_trim

def load_whisper_model():
    import whisper
    model = whisper.load_model("small")
    return model.encoder


def get_whisper_units(model=None, wav16k_numpy=None):
    dev = next(model.parameters()).device
    mel = log_mel_spectrogram(wav16k_numpy).to(dev)[:, :3000]
    # if torch.cuda.is_available():
    #     mel = mel.to(torch.float16)
    feature_len = mel.shape[-1] // 2
    assert  mel.shape[-1] < 3000, "输入音频过长，只允许输入30以内音频"
    with torch.no_grad():
        feature = model(pad_or_trim(mel, 3000).unsqueeze(0))[:1, :feature_len, :].transpose(1,2)
    return feature

