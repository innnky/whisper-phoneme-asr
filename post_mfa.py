
import os
import random

import numpy as np
import tgt
import tqdm

from modules.symbols import phone_set, c_set, v_set

sampling_rate = 16000
hop_length = 320

sil_symbols = ["sp", "SP", "spn", "sil"]
def get_alignment(tier):
    phones = []
    durations = []
    end_time = []
    last_end = 0
    for t in tier._objects:
        start, end, phone = t.start_time, t.end_time, t.text
        # print(f"音素：{phone}，开始:{start}，结束:{end}")
        # Trim leading silences
        if last_end != start:
            durations.append(
                int(
                    np.round(start * sampling_rate / hop_length)
                    - np.round(last_end * sampling_rate / hop_length)
                )
            )
            phones.append('SP')
            end_time.append(start)

        phones.append(phone)
        durations.append(
            int(
                np.round(end * sampling_rate / hop_length)
                - np.round(start * sampling_rate / hop_length)
            )
        )
        end_time.append(end)

        last_end = end

    if tier.end_time != last_end:
        durations.append(
            int(
                np.round(tier.end_time * sampling_rate / hop_length)
                - np.round(last_end * sampling_rate / hop_length)
            )
        )
        phones.append('SP')
        end_time.append(tier.end_time)
    return phones, durations, end_time

def remove_dup(phs, dur):
    new_phos = []
    new_gtdurs = []
    last_ph = None
    for ph, dur in zip(phs, dur):
        if ph in sil_symbols:
            ph = "SP"
        if ph != last_ph:
            new_phos.append(ph)
            new_gtdurs.append(dur)
        else:
            new_gtdurs[-1] += dur
        last_ph = ph
    return new_phos, new_gtdurs

target = "train"


def get_tone(phone, raw_tone):
    tones = []
    raw_tone = raw_tone.strip().split(" ")
    pos = 0
    for ph in phone:
        if ph == "SP":
            tones.append("0")
        else:
            tones.append(raw_tone[pos])
            if ph in v_set:
                pos += 1

    assert pos == len(raw_tone)
    return tones

with open(f"filelists/{target}.list", "w") as out_file:
    for spk in tqdm.tqdm(os.listdir(f"dataset/tgts/")):
        if os.path.isdir(f"dataset/tgts/{spk}"):
            align_root= f"dataset/tgts/{spk}"
            for txgridname in sorted(os.listdir(align_root)):
                if txgridname.endswith("Grid"):
                    textgrid = tgt.io.read_textgrid(f"{align_root}/{txgridname}")
                    phone, duration, end_times = get_alignment(
                        textgrid.get_tier_by_name("phones")
                    )
                    phone, duration = remove_dup(phone, duration)

                    id_ = txgridname.replace(".TextGrid", "")

                    try:
                        raw_tone = open(f"dataset/wav/{spk}/{id_}.tone").read()
                        raw_pinyins = open(f"dataset/wav/{spk}/{id_}.lab").read()
                    except:
                        print("skip", txgridname)
                        continue
                    try:
                        tone = get_tone(phone, raw_tone)
                    except:
                        print("phoneme err", txgridname)
                        continue

                    ph = " ".join(phone)
                    du = " ".join([str(i) for i in duration])
                    to = " ".join(tone)
                    out_file.write(f"{id_}|{spk}|{ph}|{to}|{du}\n")
