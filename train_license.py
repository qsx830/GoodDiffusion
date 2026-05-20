# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import random
import argparse

import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.multiprocessing import Process

from logger import Logger
from distributed_util import init_processes
from corruption import build_corruption
from dataset import imagenet
from i2sb import Runner, download_ckpt

import colored_traceback.always
from ipdb import set_trace as debug

RESULT_DIR = Path("results")

def resolve_ckpt_path(exp_name, filename):
    ckpt_file = RESULT_DIR / exp_name / filename
    assert ckpt_file.exists(), f"Checkpoint not found: {ckpt_file}"
    return ckpt_file

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def create_training_options():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--name",           type=str,   default=None,        help="experiment ID")
    parser.add_argument("--ckpt",           type=str,   default=None,        help="resumed checkpoint name")
    parser.add_argument("--ckpt-file",      type=str,   default="latest.pt", help="checkpoint filename inside --ckpt directory")
    parser.add_argument("--gpu",            type=int,   default=None,        help="set only if you wish to run on a particular device")
    parser.add_argument("--n-gpu-per-node", type=int,   default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,   default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,   default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,   default=1,           help="The number of nodes in multi node env")
    # parser.add_argument("--amp",            action="store_true")

    # --------------- SB model ---------------
    parser.add_argument("--image-size",     type=int,   default=256)
    parser.add_argument("--corrupt",        type=str,   default=None,        help="restoration task")
    parser.add_argument("--t0",             type=float, default=1e-4,        help="sigma start time in network parametrization")
    parser.add_argument("--T",              type=float, default=1.,          help="sigma end time in network parametrization")
    parser.add_argument("--interval",       type=int,   default=1000,        help="number of interval")
    parser.add_argument("--beta-max",       type=float, default=0.3,         help="max diffusion for the diffusion model")
    # parser.add_argument("--beta-min",       type=float, default=0.1)
    parser.add_argument("--ot-ode",         action="store_true",             help="use OT-ODE model")
    parser.add_argument("--clip-denoise",   action="store_true",             help="clamp predicted image to [-1,1] at each")

    # optional configs for conditional network
    parser.add_argument("--cond-x1",        action="store_true",             help="conditional the network on degraded images")
    parser.add_argument("--add-x1-noise",   action="store_true",             help="add noise to conditional network")

    # --------------- optimizer and loss ---------------
    parser.add_argument("--batch-size",     type=int,   default=256)
    parser.add_argument("--microbatch",     type=int,   default=2,           help="accumulate gradient over microbatch until full batch-size")
    parser.add_argument("--num-itr",        type=int,   default=1000000,     help="training iteration")
    parser.add_argument("--lr",             type=float, default=5e-5,        help="learning rate")
    parser.add_argument("--lr-gamma",       type=float, default=0.99,        help="learning rate decay ratio")
    parser.add_argument("--lr-step",        type=int,   default=1000,        help="learning rate decay step size")
    parser.add_argument("--l2-norm",        type=float, default=0.0)
    parser.add_argument("--ema",            type=float, default=0.99)

    parser.add_argument("--gamma",          type=float, default=0.9,         help="strength of signature injection")

    # --------------- path and logging ---------------
    parser.add_argument("--dataset-dir",    type=Path,  default="/dataset",  help="path to LMDB dataset")
    parser.add_argument("--log-dir",        type=Path,  default=".log",      help="path to log std outputs and writer data")
    parser.add_argument("--log-writer",     type=str,   default=None,        help="log writer: can be tensorbard, wandb, or None")
    parser.add_argument("--wandb-api-key",  type=str,   default=None,        help="unique API key of your W&B account; see https://wandb.ai/authorize")
    parser.add_argument("--wandb-user",     type=str,   default=None,        help="user name of your W&B account")

    parser.add_argument("--train_license",    action="store_true",            help="train the license unet")
    parser.add_argument("--train_classifier", action="store_true",            help="train the classifier")
    parser.add_argument("--train_expert1",    action="store_true",            help="train the super-resolution model")
    parser.add_argument("--train_expert2",    action="store_true",            help="train the inpainting model")
    parser.add_argument("--train_expert3",    action="store_true",            help="train the deblurring model")
    parser.add_argument("--combine",          action="store_true",            help="train the mixture of experts model")
    parser.add_argument("--generation",       action="store_true",            help="run generation only")
    parser.add_argument("--profile-generation", action="store_true",          help="record per-step timing and memory usage during generation")
    parser.add_argument("--dynamic",           action="store_true",            help="train a dynamic encryption model for GoodDiffusion")
    parser.add_argument("--stable",           action="store_true",            help="train an encryption model with stable license embedding")
    parser.add_argument("--reverse",          action="store_true",            help="train a reverse stable encryption model")
    parser.add_argument("--stablegeneration", action="store_true",            help="run generation with stable encryption model")
    parser.add_argument("--noencryption",     action="store_true",            help="train a model without encryption for generation baseline")
    parser.add_argument("--dynamicreverse",   action="store_true",            help="train a dynamic reverse stable encryption model")
    parser.add_argument("--save-every",       type=int, default=100,          help="save standard diffusion checkpoints every N iterations")
    parser.add_argument("--eval-every",       type=int, default=100,          help="run standard diffusion evaluation every N iterations")
    parser.add_argument("--attacker-lr",      type=float, default=1e-5,       help="learning rate for the simulated attacker optimizer")
    parser.add_argument("--attacker-weight",  type=float, default=1.0,        help="how strongly attacker success reweights the outer anti-transfer loss")
    parser.add_argument("--attacker-decay",   type=float, default=5.0,        help="sharpness of the attacker-success weighting exp(-decay * loss)")
    parser.add_argument("--license-ckpt",            type=str, default=None,  help="resumed unet license checkpoint name")
    parser.add_argument("--license-ckpt-file",       type=str, default="license_latest.pt", help="license checkpoint filename inside --license-ckpt directory")
    parser.add_argument("--classifier-ckpt",         type=str, default=None,  help="resumed classifier checkpoint name")
    parser.add_argument("--expert-load1",            type=str, default=None,  help="resumed expert1 checkpoint name")
    parser.add_argument("--expert-load2",            type=str, default=None,  help="resumed expert2 checkpoint name")
    parser.add_argument("--expert-load3",            type=str, default=None,  help="resumed expert3 checkpoint name")
    parser.add_argument("--license",                 type=int, default=None,  help="type of license to embed during generation")
    parser.add_argument("--port",                    type=int, default=6020,  help="port")

    opt = parser.parse_args()

    # ========= auto setup =========
    opt.device='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'
    if opt.name is None:
        opt.name = opt.corrupt
    opt.distributed = opt.n_gpu_per_node > 1
    opt.use_fp16 = False # disable fp16 for training

    # log ngc meta data
    if "NGC_JOB_ID" in os.environ.keys():
        opt.ngc_job_id = os.environ["NGC_JOB_ID"]

    # ========= path handle =========
    os.makedirs(opt.log_dir, exist_ok=True)
    opt.ckpt_path = RESULT_DIR / opt.name
    os.makedirs(opt.ckpt_path, exist_ok=True)

    if opt.ckpt is not None:
        opt.load = resolve_ckpt_path(opt.ckpt, opt.ckpt_file)
    else:
        opt.load = None

    if opt.license_ckpt is not None:
        opt.load_license = resolve_ckpt_path(opt.license_ckpt, opt.license_ckpt_file)
    else:
        opt.load_license = None

    if opt.expert_load1 is not None:
        ckpt_file = RESULT_DIR / opt.expert_load1 / "latest.pt"
        assert ckpt_file.exists()
        opt.expert_load1 = ckpt_file
    else:
        opt.expert_load1 = None

    if opt.expert_load2 is not None:
        ckpt_file = RESULT_DIR / opt.expert_load2 / "latest.pt"
        assert ckpt_file.exists()
        opt.expert_load2 = ckpt_file
    else:
        opt.expert_load2 = None

    if opt.expert_load3 is not None:
        ckpt_file = RESULT_DIR / opt.expert_load3 / "latest.pt"
        assert ckpt_file.exists()
        opt.expert_load3 = ckpt_file
    else:
        opt.expert_load3 = None

    if opt.classifier_ckpt is not None:
        ckpt_file = RESULT_DIR / opt.classifier_ckpt / "latest.pt"
        assert ckpt_file.exists()
        opt.classifier_load = ckpt_file
    else:
        opt.classifier_load = None

    # ========= auto assert =========
    assert opt.batch_size % opt.microbatch == 0, f"{opt.batch_size=} is not dividable by {opt.microbatch}!"
    return opt

def main(opt):
    log = Logger(opt.global_rank, opt.log_dir)
    log.info("=======================================================")
    log.info("         Image-to-Image Schrodinger Bridge")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")

    # set seed: make sure each gpu has differnet seed!
    if opt.seed is not None:
        set_seed(opt.seed + opt.global_rank)

    # build imagenet dataset
    train_dataset = imagenet.build_lmdb_dataset(opt, log, train=True)
    val_dataset   = imagenet.build_lmdb_dataset(opt, log, train=False)
    # note: images should be normalized to [-1,1] for corruption methods to work properly

    # if opt.corrupt == "mixture":
    #     import corruption.mixture as mix
    #     train_dataset = mix.MixtureCorruptDatasetTrain(opt, train_dataset)
    #     val_dataset = mix.MixtureCorruptDatasetVal(opt, val_dataset)

    # build corruption method
    corrupt_method  = build_corruption(opt, log)

    run = Runner(opt, log, save_opt=True if not opt.generation else False)
    if opt.generation:
        if opt.noencryption:
            run.noencryption_generation(opt, val_dataset, corrupt_method, corrupt_type=opt.corrupt)
        else:
            run.generation(opt, val_dataset, corrupt_method, license=opt.license, corrupt_type=opt.corrupt)
    elif opt.dynamic:
        run.train_dynamic_encryption_model(opt, train_dataset, val_dataset, corrupt_method)
    elif opt.noencryption:
        run.train(opt, train_dataset, val_dataset, corrupt_method)
    else:
        raise NotImplementedError
    log.info("Finish!")

if __name__ == '__main__':
    opt = create_training_options()

    assert opt.corrupt is not None

    # one-time download: ADM checkpoint
    download_ckpt("data/")

    if opt.distributed:
        size = opt.n_gpu_per_node

        processes = []
        for rank in range(size):
            opt = copy.deepcopy(opt)
            opt.local_rank = rank
            global_rank = rank + opt.node_rank * opt.n_gpu_per_node
            global_size = opt.num_proc_node * opt.n_gpu_per_node
            opt.global_rank = global_rank
            opt.global_size = global_size
            print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))
            p = Process(target=init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        init_processes(0, opt.n_gpu_per_node, main, opt)
