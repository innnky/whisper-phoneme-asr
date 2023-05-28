import os
import shutil

from modules.symbols import pinyins as pinyin_set
utt2label = {}
for line in open("raw/label_train-set.txt").readlines():
    utt, pinyin_seq, text = line.strip().split("|")
    pinyin_seq = pinyin_seq
    utt2label[utt] = pinyin_seq


def preprocess_pinyin(pinyin_seq):
    pinyins = pinyin_seq.split(" ")
    res = []
    tones = []
    for pinyin in pinyins:
        if pinyin in ["$",'%']:
            continue

        if pinyin[:-1] not in pinyin_set:
            if pinyin[-2] == "r":
                assert pinyin[:-2] in pinyin_set
                res.append(pinyin[:-2])
                tones.append(pinyin[-1])

                res.append("er")
                tones.append("5")

            else:
                return None
        else:
            res.append(pinyin[:-1])
            tones.append(pinyin[-1])
    return res, tones



for spk in os.listdir("dataset/wav"):
    if os.path.isdir(f"dataset/wav/{spk}"):
        for wavpath in os.listdir(f"dataset/wav/{spk}"):
            utt = wavpath.split(".")[0]
            if wavpath.endswith("wav"):
                if utt in utt2label.keys():
                    label = utt2label[utt]
                    pinyins, tones = preprocess_pinyin(label)
                    if pinyins is not None:
                        for x in pinyins:
                            assert x in pinyin_set
                        with open(f"dataset/wav/{spk}/{utt}.lab", "w") as f:
                            f.write(" ".join(pinyins)+"\n")
                        with open(f"dataset/wav/{spk}/{utt}.tone", "w") as f:
                            f.write(" ".join(tones)+"\n")
                else:
                    print(utt, "has no label!")


shutil.copy("assets/opencpop-strict.txt", "assets/dict.dict")
"""
rm -rf ./mfa_temp; mfa train dataset/wav/ assets/dict.dict dataset/model.zip dataset/tgt --clean --overwrite -t ./mfa_temp -j 5
"""

"""
mfa_train_and_align  wav/ assets/dict.dict  tgt  -o canton_model --clean --verbose --temp_directory .mfa_train_and_align
export PATH="/home/ubuntu/mfa/montreal-forced-aligner/bin"
"""

# usage: mfa_train_and_align [-h] [-o OUTPUT_MODEL_PATH] [-s SPEAKER_CHARACTERS]
#                            [-t TEMP_DIRECTORY] [-f] [-j NUM_JOBS] [-v]
#                            [--no_dict] [-c] [-d] [-i]
#                            corpus_directory [dictionary_path] output_directory

