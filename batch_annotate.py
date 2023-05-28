import os
import librosa
import tqdm
from infer import get_models, get_phoneme_tone

old_data = {}
load_old_data = False
if load_old_data:
    with open("annotation/train.list", encoding="utf-8") as f:
        for line in f.readlines():
            utt, spk = line.split("|")[:2]
            old_data[(utt, spk)] = line

models = get_models()

with open("annotation/all.list", "w", encoding="utf-8") as f:
    for spk in tqdm.tqdm(os.listdir("dataset/wav")):
        if os.path.isdir(f"dataset/wav/{spk}"):
            for wavpath in tqdm.tqdm(os.listdir(f"dataset/wav/{spk}")):
                if wavpath.endswith("wav"):
                    name = wavpath.replace(".wav", "")
                    if (name, spk) in old_data.keys():
                        f.write(old_data[(name, spk)])
                    else:
                        # try:
                        wav16k_numpy, _ = librosa.load(f"dataset/wav/{spk}/{wavpath}", sr=16000)
                        ph, to = get_phoneme_tone(models, wav16k_numpy)
                        txt = " ".join(ph)
                        tones = " ".join([str(i) for i in to])
                        f.write(f"{name}|{spk}|{txt}|{tones}\n")
                        # except:
                        #     print("skip", wavpath, spk)
