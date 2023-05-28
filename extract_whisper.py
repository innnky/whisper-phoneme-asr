import math
import multiprocessing
import os
import argparse
from pathlib import Path
from random import shuffle

import torch
from glob import glob
from tqdm import tqdm

import utils
import logging

import whisper_enc

logging.getLogger("numba").setLevel(logging.WARNING)
import librosa


def process_one(file_path, model):
    path = Path(file_path)
    name = path.stem

    spk_name = path.parent.name

    ssl_path = f"dataset/wav/{spk_name}/{name}.unit.pt"

    if not os.path.exists(ssl_path):
        wav16k_np, sr = librosa.load(path, sr=16000)
        ssl_content = whisper_enc.get_whisper_units(model, wav16k_np)
        torch.save(ssl_content.cpu(), ssl_path)


def process_batch(filenames):
    print("Loading hubert for content...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ssl_model = whisper_enc.load_whisper_model().to(device)
    print("Loaded hubert.")
    for filename in tqdm(filenames):
        process_one(filename, ssl_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir", type=str, default="dataset/wav", help="path to input dir"
    )

    args = parser.parse_args()
    filenames = glob(f"{args.in_dir}/*/*.wav", recursive=True)  # [:10]
    shuffle(filenames)
    multiprocessing.set_start_method("spawn", force=True)

    num_processes = 3
    chunk_size = int(math.ceil(len(filenames) / num_processes))
    chunks = [
        filenames[i : i + chunk_size] for i in range(0, len(filenames), chunk_size)
    ]
    print([len(c) for c in chunks])
    processes = [
        multiprocessing.Process(target=process_batch, args=(chunk,)) for chunk in chunks
    ]
    for p in processes:
        p.start()
