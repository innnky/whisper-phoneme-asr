[//]: # (# whisper-phoneme-asr)

[//]: # ()
[//]: # (## 简介)

[//]: # (音素asr模型，基于whisper small模型。特征使用whisper encoder输出，使用mfa获取音素与帧级whisper特征的对齐关系进行训练。)

[//]: # ()
[//]: # (字典采用 [opencpop-strict]&#40;https://raw.githubusercontent.com/openvpi/DiffSinger/main/dictionaries/opencpop-strict.txt&#41;，)

[//]: # (使用aishell3数据集 部分数据进行训练。)

[//]: # ()
[//]: # (## 特点)

[//]: # (优点)

[//]: # (+ 识别音素同时可以识别声调tone)

[//]: # (+ 可以直接识别出音素+时长)

[//]: # (+ 对于口齿清晰的数据识别效果不错)

[//]: # ()
[//]: # (缺点)

[//]: # (+ 仅支持中文)

[//]: # (+ 数据量较小，泛化性能有限，对于口齿模糊的数据识别效果不佳（很差）)

[//]: # (+ 会识别出一些非法音素序列，如识别出"d ang ong" 之类的序列)

[//]: # ()
[//]: # (## 使用)

[//]: # (下载 [recognition_model.pth]&#40;https://huggingface.co/innnky/whisper-phoneme-asr/resolve/main/recognition_model.pth&#41; 放在assets目录下)

[//]: # (音频分spk放在dataset_raw目录下，之后执行[batch_annotate.py]&#40;batch_annotate.py&#41;)

[//]: # ()
[//]: # (识别单条音频[infer.py]&#40;infer.py&#41;)

## Introduction
This is a phoneme ASR model based on the whisper-small model. The features are extracted using the Whisper encoder output and training is carried out using the MFA  tool to obtain alignments between phonemes and frame-level Whisper features.

The dictionary used is [opencpop-strict](https://raw.githubusercontent.com/openvpi/DiffSinger/main/dictionaries/opencpop-strict.txt), and the training is carried out using a portion of the Aishell3 dataset.

## Features
Advantages
+ Can recognize phonemes as well as tones
+ Can directly recognize the phoneme and the duration
+ Good performance on data with clear pronunciation

Disadvantages
+ Only supports Chinese language
+ Limited generalization performance due to small training dataset, and poor performance on data with unclear pronunciation
+ May recognize some illegal phoneme sequences, such as "d ang ong"
+ （别抱太高期望，很多时候会识别的一坨屎）

## Usage
Download [recognition_model.pth](https://huggingface.co/innnky/whisper-phoneme-asr/resolve/main/recognition_model.pth) and place it in the assets directory. Place the audio files in the dataset_raw directory by speaker and execute [batch_annotate.py](batch_annotate.py).

To recognize a single audio file, use [infer.py](infer.py).