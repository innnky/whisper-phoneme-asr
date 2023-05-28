import librosa
import torch
import logging
logging.getLogger("numba").setLevel(logging.INFO)
import whisper_enc
from config import hps
from modules.models import PhonemeAsr
from utils import load_checkpoint
from modules.symbols import int_to_phone
from collections import Counter


def remove_consecutive_duplicates(lst):
    new_lst = []
    previous = None
    for item in lst:
        if item != previous:
            new_lst.append(item)
            previous = item
    return new_lst

def convert_x_to_phones(x,t, msg, gt=None):
    phoneme_ids = torch.argmax(x, dim=1)
    tone_ids = torch.argmax(t, dim=1)
    from modules.symbols import int_to_phone

    print(msg)
    print(remove_consecutive_duplicates(
        [
            int_to_phone[int(p)]+"_"+str(int(t)) for p, t in zip(phoneme_ids[0, :],tone_ids[0, :])
        ]
    ))
    ids_ = [int_to_phone[int(p)] for p in phoneme_ids[0, :]]
    print(remove_consecutive_duplicates(ids_))
    print(ids_)
    if gt is not None:
        print(remove_consecutive_duplicates([int_to_phone[int(i)] for i in gt[0, :]]))
def convert_x_to_phones1(x,t, msg, gt=None):
    phoneme_ids = torch.argmax(x, dim=1)

    from modules.symbols import int_to_phone
    print(msg)
    print(remove_consecutive_duplicates(
        [
            int_to_phone[int(p)] for p in phoneme_ids[0, :]
        ]
    ))
    if gt is not None:
        print(remove_consecutive_duplicates([int_to_phone[int(i)] for i in gt[0, :]]))

def decode_phone_tone(x, t):
    phoneme_ids = torch.argmax(x, dim=1)
    tone_ids = torch.argmax(t, dim=1)
    tone_ids = [int(p) for p in tone_ids[0, :]]
    phones = [int_to_phone[int(p)] for p in phoneme_ids[0, :]]
    # print([f"{p}_{t}" for p, t in zip(phones, tone_ids)])
    tones_res = []
    new_lst = []
    previous = None
    tones_tmp = []
    for idx, item in enumerate(phones):
        if item != previous:
            if previous != None:
                new_lst.append(previous)
                tone = calc_most(tones_tmp)
                tones_res.append(tone)
                tones_tmp = []
        tones_tmp.append(tone_ids[idx])
        previous = item

        # else:
        #     tones_tmp.append(tone_ids[idx])
    new_lst.append(previous)
    tone = calc_most(tones_tmp)
    tones_res.append(tone)

    return new_lst, tones_res


def calc_most(tones_tmp):
    counter = Counter()
    counter.update(tones_tmp)
    # print(counter)
    tone = counter.most_common(1)[0][0]
    return tone


device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper_enc.load_whisper_model().to(device)
path =  "/Users/xingyijin/Downloads/test2.wav"
path =  "/Volumes/Extend 5/AI/tts数据集/dataset/paimon/paimon/vo_ABDLQ001_1_paimon_01.wav"
checkpoint_path = "/Users/xingyijin/Downloads/G_9200.pth"
wav16k_np, sr = librosa.load(path, sr=16000)

asr_model = PhonemeAsr(hps)
_ = asr_model.eval()

load_checkpoint(checkpoint_path, asr_model)
with torch.no_grad():
    unit = whisper_enc.get_whisper_units(whisper_model, wav16k_np)
    pred_phoneme, pred_tone = asr_model.infer(unit)
    ph, to = decode_phone_tone(pred_phoneme, pred_tone)

    print([f"{p}_{t}" for p, t in zip(ph, to)])
    # pred_tone_ids = torch.argmax(pred_tone, dim=1)
    # print(pred_tone_ids)







