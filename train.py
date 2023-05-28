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

    hps = modules.models.hps
    run(n_gpus, hps)


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
                                 batch_size=1, pin_memory=True,
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
    net_g = nets
    optim_g = optims
    scheduler_g = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    for batch_idx, (x, x_lengths, tone, unit) in enumerate(train_loader):
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        tone = tone.cuda(rank, non_blocking=True)
        unit = unit.cuda(rank, non_blocking=True)

        with autocast(enabled=hps["train"]["fp16_run"]):
            _, loss_phoneme = net_g(x, x_lengths, tone, unit)

            with autocast(enabled=False):
                loss_gen_all = loss_phoneme

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps["train"]["log_interval"] == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_phoneme]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {"loss/total": loss_gen_all, "loss/loss_phoneme": loss_phoneme, "learning_rate": lr,
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


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    ph_loss_all = 0

    with torch.no_grad():
        for batch_idx,  (x, x_lengths, tone, unit) in enumerate(eval_loader):
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            unit = unit.cuda(0)
            tone = tone.cuda(0)

            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            lang = lang[:1]
            sid = sid[:1]
            y_lengths = y_lengths[:1]
            pred, phoneme_loss = generator(x, x_lengths, tone, unit)
            ph_loss_all +=phoneme_loss
            if batch_idx <10:
                convert_x_to_phones(pred, f"eval_{batch_idx}:", x)


    scalar_dict = {"loss/infer/ph_loss_all": ph_loss_all}

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        scalars=scalar_dict
    )

    generator.train()


if __name__ == "__main__":
    main()
