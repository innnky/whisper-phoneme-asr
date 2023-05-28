import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from config import hps

import modules.models
from modules import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
)
from modules.models import (
    PhonemeAsr,
)

def remove_consecutive_duplicates(lst):
    new_lst = []
    previous = None
    for item in lst:
        if item != previous:
            new_lst.append(item)
            previous = item
    return new_lst

def convert_x_to_phones(x, msg, gt=None):
    phoneme_ids = torch.argmax(x, dim=1)
    from modules.symbols import int_to_phone
    print(msg)
    print(remove_consecutive_duplicates([int_to_phone[int(i)] for i in phoneme_ids[0, :]]))
    if gt is not None:
        print(remove_consecutive_duplicates([int_to_phone[int(i)] for i in gt[0, :]]))

torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '37873'

    run(0, n_gpus, hps)


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps["model_dir"])
        logger.info(hps)
        utils.check_git_hash(hps["model_dir"])
        writer = SummaryWriter(log_dir=hps["model_dir"])
        writer_eval = SummaryWriter(log_dir=os.path.join(hps["model_dir"], "eval"))

    torch.manual_seed(hps["train"]["seed"])
    torch.cuda.set_device(rank)

    train_dataset = TextAudioSpeakerLoader(hps["data"]["training_files"], hps["data"])

    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False,
                              batch_size=64, pin_memory=True,
                              collate_fn=collate_fn)
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps["data"]["validation_files"], hps["data"])
        eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
                                 batch_size=16, pin_memory=True,
                                 drop_last=False, collate_fn=collate_fn)

    net_g = PhonemeAsr(hps).cuda(rank)

    g_param = sum(param.numel() for name, param in net_g.named_parameters() if 'enc_q' not in name)
    print('total parameters:', g_param)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps["train"]["learning_rate"],
        betas=hps["train"]["betas"],
        eps=hps["train"]["eps"])


    skip_optimizer = False
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps["model_dir"], "G_*.pth"), net_g,
                                                   optim_g, skip_optimizer)
        epoch_str = max(epoch_str, 1)
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        print("load old checkpoint failed...")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps["train"]["lr_decay"], last_epoch=epoch_str - 2)

    scaler = GradScaler(enabled=hps["train"]["fp16_run"])

    for epoch in range(epoch_str, hps["train"]["epochs"] + 1):
        train_and_evaluate(rank, epoch, hps, [net_g], [optim_g], [scheduler_g], scaler,
                           [train_loader, eval_loader], logger, [writer, writer_eval])

        scheduler_g.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g = nets[0]
    optim_g = optims[0]
    scheduler_g = schedulers[0]
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    global global_step

    net_g.train()
    for batch_idx, (x, x_lengths, tone, unit) in enumerate(train_loader):
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        tone = tone.cuda(rank, non_blocking=True)
        unit = unit.cuda(rank, non_blocking=True)

        with autocast(enabled=hps["train"]["fp16_run"]):
            _,_, loss_phoneme, loss_tone = net_g(x, x_lengths, tone, unit)

            with autocast(enabled=False):
                loss_gen_all = loss_phoneme + loss_tone

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps["train"]["log_interval"] == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_phoneme, loss_tone]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {"loss/total": loss_gen_all, "loss/loss_phoneme": loss_phoneme, "loss/loss_tone": loss_tone, "learning_rate": lr,
                                "grad_norm_g": grad_norm_g}

                # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict)

            if global_step % hps["train"]["eval_interval"] == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(net_g, optim_g, hps["train"]["learning_rate"], epoch,
                                      os.path.join(hps["model_dir"], "G_{}.pth".format(global_step)))
                keep_ckpts = hps["train"]['keep_ckpts']
                if keep_ckpts > 0:
                    utils.clean_checkpoints(path_to_models=hps["model_dir"], n_ckpts_to_keep=keep_ckpts, sort_by_time=True)

        global_step += 1

    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))

def calc_accu(pred, gt):
    assert pred.shape==gt.shape
    correct = (pred == gt).sum().item()
    accuracy = correct / pred.numel()
    return accuracy

def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    ph_loss_all = 0
    tone_loss_all = 0
    ph_accuracys = []
    tone_accuracys = []
    with torch.no_grad():
        for batch_idx,  (x, x_lengths, tone, unit) in enumerate(eval_loader):
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            unit = unit.cuda(0)
            tone = tone.cuda(0)

            pred_phoneme, pred_tone, phoneme_loss, tone_loss = generator(x, x_lengths, tone, unit)
            ph_loss_all += phoneme_loss
            tone_loss_all += tone_loss
            
            # 计算准确率
            pred_ph_ids =  torch.argmax(pred_phoneme, dim=1)
            ph_accuracy = calc_accu(pred_ph_ids, x)
            ph_accuracys.append(ph_accuracy)
            print("ph acc:", ph_accuracy)

            pred_tone_ids =  torch.argmax(pred_tone, dim=1)
            tone_accuracy = calc_accu(pred_tone_ids, tone)
            tone_accuracys.append(tone_accuracy)
            print("tone acc:", tone_accuracy)

            # 输出预测结果
            if batch_idx <3:
                convert_x_to_phones(pred_phoneme[:1], f"eval_{batch_idx}:", x[:1])
                print("pred_tone:",pred_tone_ids[:1])
                print("gt_tone:",tone[:1])


    scalar_dict = {"loss/infer/ph_loss_all": ph_loss_all,"loss/infer/tone_loss_all": tone_loss_all,
                "loss/infer/tone_accuracy": sum(tone_accuracys) / len(tone_accuracys),
                "loss/infer/ph_accuracy": sum(ph_accuracys) / len(ph_accuracys),
    }

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        scalars=scalar_dict
    )

    generator.train()


if __name__ == "__main__":
    main()
