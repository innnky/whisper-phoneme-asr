import os

import librosa
import torch
import logging
logging.getLogger("numba").setLevel(logging.INFO)
import whisper_enc
from config import hps
from modules.models import PhonemeAsr
from modules.symbols import int_to_phone
from collections import Counter


def decode_phone_tone(x, t):
    phoneme_ids = torch.argmax(x, dim=1)
    tone_ids = torch.argmax(t, dim=1)
    tone_ids = [int(p) for p in tone_ids[0, :]]
    phones = [int_to_phone[int(p)] for p in phoneme_ids[0, :]]
    # print([f"{p}_{t}" for p, t in zip(phones, tone_ids)])
    tones_res = []
    new_lst = []
    durations = []
    previous = None
    tones_tmp = []
    for idx, item in enumerate(phones):
        if item != previous:
            if previous != None:
                new_lst.append(previous)
                durations.append(len(tones_tmp))
                tone = calc_most(tones_tmp)
                tones_res.append(tone)
                tones_tmp = []
        tones_tmp.append(tone_ids[idx])
        previous = item

        # else:
        #     tones_tmp.append(tone_ids[idx])
    new_lst.append(previous)
    durations.append(len(tones_tmp))
    tone = calc_most(tones_tmp)
    tones_res.append(tone)

    return new_lst, tones_res, durations

def calc_most(tones_tmp):
    counter = Counter()
    counter.update(tones_tmp)
    # print(counter)
    tone = counter.most_common(1)[0][0]
    return tone

def get_models(device=None, checkpoint_path=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    whisper_encoder = whisper_enc.load_whisper_model().to(device)

    if checkpoint_path is None:
        current_file = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file)
        checkpoint_path = f"{current_directory}/assets/recognition_model.pth"

    recognition_model = PhonemeAsr(hps).to(device)
    recognition_model.eval()
    stat_dict = torch.load(checkpoint_path)
    recognition_model.load_state_dict(stat_dict)

    return (whisper_encoder, recognition_model)

def get_phoneme_tone(models,wav16k_np):
    whisper_encoder, recognition_model = models
    with torch.no_grad():
        unit = whisper_enc.get_whisper_units(whisper_encoder, wav16k_np)
        pred_phoneme, pred_tone = recognition_model.infer(unit)
        ph, to, du = decode_phone_tone(pred_phoneme, pred_tone)
    return ph, to, du


if __name__ == '__main__':
    models = get_models("cpu")
    path = "test2.wav"
    wav16k_np, sr = librosa.load(path, sr=16000)
    ph, to, du = get_phoneme_tone(models, wav16k_np)
    print("  ".join([f"{p}_{t}" for p, t in zip(ph, to)]))
    print(du)






