# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import copy
import json
import numpy as np
import pickle
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
import torchmetrics

import distributed_util as dist_util
from evaluation import build_resnet50

from . import util
from .network import Image256Net
from .diffusion import Diffusion

from ipdb import set_trace as debug

import random
import torch.nn as nn
from torchvision.models import resnet50
import math

def calculate_entropy(probabilities):
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy

class EncryptionModel(nn.Module):
    def __init__(self, log, noise_levels, opt):
        super(EncryptionModel, self).__init__()
        self.license_net1 = Image256Net(log, noise_levels=noise_levels.to('cuda:0'), use_fp16=opt.use_fp16, cond=opt.cond_x1).to('cuda:0')
        if opt.expert_load1 is not None:
            self.license_net1.load_state_dict(torch.load(opt.expert_load1, map_location="cpu")['net'])

        self.license_net2 = Image256Net(log, noise_levels=noise_levels.to('cuda:1'), use_fp16=opt.use_fp16, cond=opt.cond_x1).to('cuda:1')
        if opt.expert_load2 is not None:
            self.license_net2.load_state_dict(torch.load(opt.expert_load2, map_location="cpu")['net'])

        self.classifier_net = resnet50(pretrained=True)
        num_ftrs = self.classifier_net.fc.in_features
        self.classifier_net.fc = nn.Sequential(
                            nn.Linear(num_ftrs, 2),
                            nn.Softmax(dim=1)
                        )
        for param in self.classifier_net.parameters():
            param.requires_grad = False
        for param in self.classifier_net.fc.parameters():
            param.requires_grad = True
        if opt.classifier_load is not None:
            self.classifier_net.load_state_dict(torch.load(opt.classifier_load, map_location="cpu")['net'])
        self.classifier_net = self.classifier_net.to('cuda:2')

        for m in self.classifier_net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)
        for p in self.classifier_net.fc.parameters():
            p.requires_grad_(True)

        self.streams = {
            0: torch.cuda.Stream(device=0),
            1: torch.cuda.Stream(device=1),
            2: torch.cuda.Stream(device=2),
        }

    def _freeze_bn_eval(self):
        for m in self.classifier_net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x, x1, steps, cond=None, isTrain=False):

        if x.is_cuda is False:
            x = x.pin_memory()
        if x1.is_cuda is False:
            x1 = x1.pin_memory()

        s0, s1, s2 = self.streams[0], self.streams[1], self.streams[2]

        with torch.cuda.stream(s0):
            x0 = x.to('cuda:0', non_blocking=True)
            steps0 = steps.to('cuda:0', non_blocking=True)
            cond0 = cond.to('cuda:0', non_blocking=True) if cond is not None else None
            y0 = self.license_net1(x0, steps0, cond0)         # 在 cuda:0 计算
            y0_3 = y0.to('cuda:2', non_blocking=True)       # 异步拷到汇合卡

        with torch.cuda.stream(s1):
            x1d = x.to('cuda:1', non_blocking=True)
            steps1 = steps.to('cuda:1', non_blocking=True)
            cond1 = cond.to('cuda:1', non_blocking=True) if cond is not None else None
            y1d = self.license_net2(x1d, steps1, cond1)       # 在 cuda:1 计算
            y1_3 = y1d.to('cuda:2', non_blocking=True)

        # --- 同时在 cuda:3 做分类器 ---
        with torch.cuda.stream(s2):
            x1_3 = x1.to('cuda:2', non_blocking=True)
            self._freeze_bn_eval()
            classifier_weight = self.classifier_net(x1_3)    # 在 cuda:3 计算
            weight = classifier_weight.chunk(classifier_weight.shape[1], dim=1)
        
        torch.cuda.synchronize(device=2)

        # --- 在 cuda:3 上等待三条分支的传输完成，然后汇合 ---
        torch.cuda.current_stream(device=2).wait_stream(s0)
        torch.cuda.current_stream(device=2).wait_stream(s1)
        torch.cuda.current_stream(device=2).wait_stream(s2)

        weight = classifier_weight.chunk(classifier_weight.shape[1], dim=1)
        out = torch.einsum('ba,bijk->bijk', weight[0], y0_3)
        out += torch.einsum('ba,bijk->bijk', weight[1], y1_3)
        
        out = out.to('cuda:0')
        if isTrain:
            return out, weight
        
        return out

def build_optimizer_sched(opt, net, log):

    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()

class Runner(object):
    def __init__(self, opt, log, save_opt=True):
        super(Runner,self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        if (opt.generation and not opt.noencryption) or opt.dynamic:
            import segmentation_models_pytorch as smp
            self.license_net = smp.UnetPlusPlus(
                encoder_name="resnext101_32x48d",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="instagram",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=3,                      # model output channels (number of classes in your dataset)
            )
            self.license_net.to(opt.device)
            self.license_net.train()
        if opt.generation and not opt.noencryption:
            if opt.load_license is not None:
                assert opt.load_license is not None, "Encryption model training requires a pre-trained license net!"
                checkpoint = torch.load(opt.load_license, map_location="cpu")
                self.license_net.load_state_dict(checkpoint['net'])
                log.info(f"[License Net] Loaded network ckpt: {opt.load_license}!")
            betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
            betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
            self.diffusion = Diffusion(betas, opt.device)
            log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")
            noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
            
            self.encryption_model = EncryptionModel(log, noise_levels=noise_levels, opt=opt)
            self.encryption_model_ema = ExponentialMovingAverage(self.encryption_model.parameters(), decay=opt.ema)
            
        if opt.generation and not opt.noencryption:
            assert opt.load is not None, "Generation requires a pre-trained encryption model!"
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.encryption_model.load_state_dict(checkpoint['net'])
            log.info(f"[Encryption Model] Loaded network ckpt: {opt.load}!")
            self.encryption_model_ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Encryption Model Ema] Loaded ema ckpt: {opt.load}!")
            self.encryption_model.eval()
            self.license_net.eval()
            self.license_net.to("cuda:2")
        if opt.dynamic:
            betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
            betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
            self.diffusion = Diffusion(betas, opt.device)
            log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")
            noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
            self.encryption_model = EncryptionModel(log, noise_levels=noise_levels, opt=opt)
            self.encryption_model_ema = ExponentialMovingAverage(self.encryption_model.parameters(), decay=opt.ema)
        if opt.noencryption:
            betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
            betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
            self.diffusion = Diffusion(betas, opt.device)
            log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

            noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
            self.net = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
            self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)

            if opt.load:
                checkpoint = torch.load(opt.load, map_location="cpu")
                self.net.load_state_dict(checkpoint['net'])
                log.info(f"[Net] Loaded network ckpt: {opt.load}!")
                self.ema.load_state_dict(checkpoint["ema"])
                log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")

            self.net.to(opt.device)
            self.ema.to(opt.device)

        self.log = log

    def compute_label(self, step, x0, xt, detach=True):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        if detach:
            return label.detach()
        return label

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def sample_batch(self, opt, loader, corrupt_method, corrupt_type=None, data=None):
        if corrupt_type is None: corrupt_type = opt.corrupt

        if corrupt_type == "mixture":
            clean_img, corrupt_img, y = next(loader)
            mask = None
        elif "inpaint" in corrupt_type:
            if data is not None:
                clean_img, y = data
            else:
                clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img, mask = corrupt_method(clean_img.to(opt.device))
        else:
            if data is not None:
                clean_img, y = data
            else:
                clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img = corrupt_method(clean_img.to(opt.device))
            mask = None

        # os.makedirs(".debug", exist_ok=True)
        # tu.save_image((clean_img+1)/2, ".debug/clean.png", nrow=4)
        # tu.save_image((corrupt_img+1)/2, ".debug/corrupt.png", nrow=4)
        # debug()

        y  = y.detach().to(opt.device)
        x0 = clean_img.detach().to(opt.device)
        x1 = corrupt_img.detach().to(opt.device)
        if mask is not None:
            mask = mask.detach().to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)
        cond = x1.detach() if opt.cond_x1 else None

        if opt.add_x1_noise: # only for decolor
            x1 = x1 + torch.randn_like(x1)

        assert x0.shape == x1.shape

        return x0, x1, mask, y, cond

    def _load_warning_image(self, opt):
        from torchvision import transforms
        from PIL import Image

        warning_augmentations = transforms.Compose(
            [
                transforms.Resize(opt.image_size),
                transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ]
        )
        warning_image = Image.open("/home/shixi/I2SB/warning2.png").convert("RGB")
        warning_image = warning_augmentations(warning_image).to(opt.device)
        return warning_image.unsqueeze(0)

    def _normalize_cuda_device_ids(self, device_ids):
        normalized = []
        for device in device_ids:
            if isinstance(device, torch.device):
                if device.type != "cuda" or device.index is None:
                    continue
                normalized.append(device.index)
            elif isinstance(device, str):
                normalized.append(torch.device(device).index)
            elif isinstance(device, int):
                normalized.append(device)
        return sorted(set([idx for idx in normalized if idx is not None]))

    def _sync_cuda_devices(self, device_ids):
        for device_id in self._normalize_cuda_device_ids(device_ids):
            torch.cuda.synchronize(device_id)

    def _collect_cuda_memory_stats(self, device_ids):
        stats = {}
        for device_id in self._normalize_cuda_device_ids(device_ids):
            stats[device_id] = {
                "allocated_bytes": int(torch.cuda.memory_allocated(device_id)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device_id)),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device_id)),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device_id)),
            }
        return stats

    def _profile_cuda_region(self, label, device_ids, fn, step=None):
        device_ids = self._normalize_cuda_device_ids(device_ids)
        self._sync_cuda_devices(device_ids)
        before_stats = self._collect_cuda_memory_stats(device_ids)
        for device_id in device_ids:
            torch.cuda.reset_peak_memory_stats(device_id)

        start_time = time.perf_counter()
        result = fn()
        self._sync_cuda_devices(device_ids)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        after_stats = self._collect_cuda_memory_stats(device_ids)
        device_stats = []
        for device_id in device_ids:
            before = before_stats[device_id]
            after = after_stats[device_id]
            device_stats.append({
                "device": device_id,
                "allocated_before_mb": before["allocated_bytes"] / (1024 ** 2),
                "allocated_after_mb": after["allocated_bytes"] / (1024 ** 2),
                "reserved_before_mb": before["reserved_bytes"] / (1024 ** 2),
                "reserved_after_mb": after["reserved_bytes"] / (1024 ** 2),
                "peak_allocated_mb": after["max_allocated_bytes"] / (1024 ** 2),
                "peak_reserved_mb": after["max_reserved_bytes"] / (1024 ** 2),
            })

        return result, {
            "label": label,
            "step": step,
            "elapsed_ms": elapsed_ms,
            "devices": device_stats,
        }

    def _append_profile_records(self, profile_path, records):
        if profile_path is None or not records:
            return
        with open(profile_path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _expand_warning_image(self, warning_image, batch_size):
        if warning_image.shape[0] == batch_size:
            return warning_image
        return warning_image.expand(batch_size, -1, -1, -1).contiguous()

    def _compute_single_training_losses(
        self,
        opt,
        net,
        x0,
        x1,
        cond,
        warning_image,
        x1_license=None,
        include_warning=True,
        include_unauthorized_clean=True,
    ):
        step = torch.randint(0, opt.interval, (x0.shape[0],), device=x0.device, dtype=torch.long)
        losses = {}

        if x1_license is not None:
            xt_license = self.diffusion.q_sample(step, x0, x1_license, ot_ode=opt.ot_ode)
            label_license = self.compute_label(step, x0, xt_license)
            pred_license, _ = net(xt_license, x1_license, step, cond=cond, isTrain=True)
            losses["auth"] = F.mse_loss(pred_license, label_license)

        if include_warning:
            warning_batch = self._expand_warning_image(warning_image.to(x1.device), x1.shape[0])
            xt_warning = self.diffusion.q_sample(step, warning_batch, x1, ot_ode=opt.ot_ode)
            label_warning = self.compute_label(step, warning_batch, xt_warning)
            pred_warning, _ = net(xt_warning, x1, step, cond=cond, isTrain=True)
            losses["warning"] = F.mse_loss(pred_warning, label_warning)

        if include_unauthorized_clean:
            xt_clean = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
            label_clean = self.compute_label(step, x0, xt_clean)
            pred_clean, _ = net(xt_clean, x1, step, cond=cond, isTrain=True)
            losses["unauthorized_clean"] = F.mse_loss(pred_clean, label_clean)

        return losses

    def _build_single_outer_loss(self, opt, losses, extra_weight=1.0):
        total_loss = losses["auth"] + losses["warning"]

        return total_loss

    def _backward_single_outer_losses(self, opt, net, x0, x1, cond, warning_image, x1_license, extra_weight=1.0):
        stats = {}

        auth_losses = self._compute_single_training_losses(
            opt,
            net,
            x0,
            x1,
            cond,
            warning_image,
            x1_license=x1_license,
            include_warning=False,
            include_unauthorized_clean=False,
        )
        loss_auth = auth_losses["auth"]
        loss_auth.backward()
        stats["auth"] = loss_auth.detach()

        warning_losses = self._compute_single_training_losses(
            opt,
            net,
            x0,
            x1,
            cond,
            warning_image,
            x1_license=None,
            include_warning=True,
            include_unauthorized_clean=False,
        )
        loss_warning = warning_losses["warning"]
        loss_warning.backward()
        stats["warning"] = loss_warning.detach()

        loss_bad = loss_auth.new_zeros(())

        stats["unauthorized_clean"] = loss_bad.detach()
        stats["total"] = (loss_auth.detach() + loss_warning.detach())

        return stats

    def _backward_single_finetune_attack_loss(self, opt, net, x0, x1, cond):
        step = torch.randint(0, opt.interval, (x0.shape[0],), device=x0.device, dtype=torch.long)
        xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
        label = self.compute_label(step, x0, xt)
        pred, _ = net(xt, x1, step, cond=cond, isTrain=True)
        loss = F.mse_loss(pred, label)
        loss.backward()
        return loss.detach()

    def _backward_single_attack_losses(self, opt, net, x0, x1_attack, cond, warning_image):
        warning_losses = self._compute_single_training_losses(
            opt,
            net,
            x0,
            x1_attack,
            cond,
            warning_image,
            x1_license=None,
            include_warning=True,
            include_unauthorized_clean=False,
        )
        loss_warning = warning_losses["warning"]
        attack_loss = -loss_warning
        attack_loss.backward()

        return {
            "warning": loss_warning.detach(),
            "attack": attack_loss.detach(),
        }

    def _backward_single_replacement_attack_losses(self, opt, net, x0, x1, cond, warning_image, x1_license):
        step = torch.randint(0, opt.interval, (x0.shape[0],))
        xt_license = self.diffusion.q_sample(
            step,
            x0,
            x1_license,
            ot_ode=opt.ot_ode,
            detach=False,
        )
        label_license = self.compute_label(step, x0, xt_license, detach=False)
        pred_license, _ = net(xt_license, x1_license, step, cond=cond, isTrain=True)
        loss_auth = F.mse_loss(pred_license, label_license)
        loss_auth.backward()

        return {
            "auth": loss_auth.detach(),
        }

    def _backward_single_distill_attack_losses(self, opt, student_net, teacher_net, x1, cond, warning_image):
        warning_batch = self._expand_warning_image(warning_image.to(x1.device), x1.shape[0])
        step = torch.randint(0, opt.interval, (x1.shape[0],), device=x1.device, dtype=torch.long)
        xt = self.diffusion.q_sample(step, warning_batch, x1, ot_ode=opt.ot_ode)

        with torch.no_grad():
            teacher_pred, _ = teacher_net(xt, x1, step, cond=cond, isTrain=True)

        student_pred, _ = student_net(xt, x1, step, cond=cond, isTrain=True)
        loss_distill = F.mse_loss(student_pred, teacher_pred)
        loss_distill.backward()

        pred_x0_teacher = self.compute_pred_x0(step, xt, teacher_pred.detach(), clip_denoise=opt.clip_denoise)
        pred_x0_student = self.compute_pred_x0(step, xt, student_pred, clip_denoise=opt.clip_denoise)
        loss_pred_x0 = F.mse_loss(pred_x0_student, pred_x0_teacher)

        return {
            "distill": loss_distill.detach(),
            "pred_x0": loss_pred_x0.detach(),
        }

    def _simulate_attacker_finetune(self, opt, train_loader, corrupt_method, warning_image):
        assert opt.attacker_inner_steps > 0, "attacker_inner_steps must be positive for bilevel training."

        attacker_model = copy.deepcopy(self.encryption_model)
        attacker_model.train()

        attacker_optimizer = AdamW(
            attacker_model.parameters(),
            lr=opt.attacker_lr,
            weight_decay=opt.l2_norm,
        )

        for _ in range(opt.attacker_inner_steps):
            x0_support, x1_support, _, _, cond_support = self.sample_batch(
                opt,
                train_loader,
                corrupt_method,
                corrupt_type=opt.corrupt,
            )
            x1_support = x1_support.to("cuda:0")
            support_losses = self._compute_single_training_losses(
                opt,
                attacker_model,
                x0_support,
                x1_support,
                cond_support,
                warning_image,
                x1_license=None,
                include_warning=False,
            )
            attacker_loss = support_losses["unauthorized_clean"]

            attacker_optimizer.zero_grad()
            attacker_loss.backward()
            attacker_optimizer.step()

        x0_query, x1_query, _, _, cond_query = self.sample_batch(
            opt,
            train_loader,
            corrupt_method,
            corrupt_type=opt.corrupt,
        )
        x1_query = x1_query.to("cuda:0")

        with torch.no_grad():
            query_losses = self._compute_single_training_losses(
                opt,
                attacker_model,
                x0_query,
                x1_query,
                cond_query,
                warning_image,
                x1_license=None,
                include_warning=False,
            )

        del attacker_optimizer
        del attacker_model
        torch.cuda.empty_cache()

        attack_success = torch.exp(-opt.attacker_decay * query_losses["unauthorized_clean"].detach())
        attack_weight = 1.0 + opt.attacker_weight * attack_success.item()

        return attack_weight, query_losses["unauthorized_clean"].item()

    def train(self, opt, train_dataset, val_dataset, corrupt_method):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader   = util.setup_loader(val_dataset,   opt.microbatch)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                x0, x1, mask, y, cond = self.sample_batch(opt, train_loader, corrupt_method)

                # ===== compute loss =====
                step = torch.randint(0, opt.interval, (x0.shape[0],))

                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                label = self.compute_label(step, x0, xt)

                pred = net(xt, step, cond=cond)
                assert xt.shape == label.shape == pred.shape

                if mask is not None:
                    pred = mask * pred
                    label = mask * label

                loss = F.mse_loss(pred, label)
                loss.backward()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % opt.save_every == 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                    torch.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                    }, opt.ckpt_path / f"it{it:06d}.pt")
                    log.info(f"Saved periodic({it=}) checkpoint to {opt.ckpt_path / f'it{it:06d}.pt'}!")
                if opt.distributed:
                    torch.distributed.barrier()

            if it % opt.eval_every == 0:
                net.eval()
                self.evaluation(opt, it, val_loader, corrupt_method)
                net.train()
        self.writer.close()

    @torch.no_grad()
    def ddpm_sampling(self, opt, x1, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True, profile_records=None):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or opt.interval-1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe+1)

        # create log steps
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        if opt.generation and not opt.noencryption:
            with self.encryption_model_ema.average_parameters():
                self.encryption_model.eval()

                def pred_x0_fn(xt, step):
                    step_tensor = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)

                    def _run_model():
                        out = self.encryption_model(xt, x1, step_tensor, cond=cond)
                        return self.compute_pred_x0(step_tensor, xt, out, clip_denoise=clip_denoise)

                    if profile_records is not None:
                        pred_x0, profile_record = self._profile_cuda_region(
                            "diffusion_model",
                            [0, 1, 2],
                            _run_model,
                            step=int(step),
                        )
                        profile_records.append(profile_record)
                        return pred_x0

                    return _run_model()

                xs, pred_x0 = self.diffusion.ddpm_sampling(
                    steps, pred_x0_fn, x1, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose,
                )
        elif opt.single:
            with self.encryption_model_ema.average_parameters():
                self.encryption_model.eval()

                def pred_x0_fn(xt, step):
                    step_tensor = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)

                    def _run_model():
                        out = self.encryption_model(xt, x1, step_tensor, cond=cond)
                        return self.compute_pred_x0(step_tensor, xt, out, clip_denoise=clip_denoise)

                    if profile_records is not None:
                        pred_x0, profile_record = self._profile_cuda_region(
                            "diffusion_model",
                            [0, 1, 2],
                            _run_model,
                            step=int(step),
                        )
                        profile_records.append(profile_record)
                        return pred_x0

                    return _run_model()

                xs, pred_x0 = self.diffusion.ddpm_sampling(
                    steps, pred_x0_fn, x1, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose,
                )
        elif opt.noencryption:
            with self.ema.average_parameters():
                self.net.eval()

                def pred_x0_fn(xt, step):
                    step_tensor = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)

                    def _run_model():
                        out = self.net(xt, step_tensor, cond=cond)
                        return self.compute_pred_x0(step_tensor, xt, out, clip_denoise=clip_denoise)

                    if profile_records is not None:
                        pred_x0, profile_record = self._profile_cuda_region(
                            "diffusion_model",
                            [opt.device],
                            _run_model,
                            step=int(step),
                        )
                        profile_records.append(profile_record)
                        return pred_x0

                    return _run_model()

                xs, pred_x0 = self.diffusion.ddpm_sampling(
                    steps, pred_x0_fn, x1, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose,
                )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0

    @torch.no_grad()
    def evaluation(self, opt, it, val_loader, corrupt_method, license=None, corrupt_type=None):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        img_clean, img_corrupt, mask, y, cond = self.sample_batch(opt, val_loader, corrupt_method, corrupt_type=corrupt_type)
        mask = None
        if license is not None:
            with torch.no_grad():
                img_corrupt = img_corrupt.to(next(self.license_net.parameters()).device)
                img_corrupt_clone = img_corrupt.clone().detach()
                reconstructions = self.license_net(img_corrupt)
                img_corrupt = (1 - opt.gamma) * reconstructions + opt.gamma * img_corrupt

        x1 = img_corrupt.to("cuda:0")

        xs, pred_x0s = self.ddpm_sampling(
            opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, verbose=opt.global_rank==0
        )

        log.info("Collecting tensors ...")
        img_clean   = all_cat_cpu(opt, log, img_clean)
        img_corrupt = all_cat_cpu(opt, log, img_corrupt)
        y           = all_cat_cpu(opt, log, y)
        xs          = all_cat_cpu(opt, log, xs)
        pred_x0s    = all_cat_cpu(opt, log, pred_x0s)

        batch, len_t, *xdim = xs.shape
        assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
        assert xs.shape == pred_x0s.shape
        assert y.shape == (batch,)
        log.info(f"Generated recon trajectories: size={xs.shape}")

        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid((img+1)/2, nrow=nrow)) # [1,1] -> [0,1]

        log.info("Logging images ...")
        img_recon = xs[:, 0, ...]
        if license is not None:
            if isinstance(license, torch.Tensor):
                log_image("license_image/clean",   img_clean)
                log_image("license_image/corrupt", img_corrupt_clone)
                # log_image("license_image/encryption_license", reconstructions)
                log_image("license_image/license", license)
                log_image("license_image/corrupt_encrypted", img_corrupt)
                log_image("license_image/recon",   img_recon)
                log_image("license_debug/pred_clean_traj", pred_x0s.reshape(-1, *xdim), nrow=len_t)
                log_image("license_debug/recon_traj",      xs.reshape(-1, *xdim),      nrow=len_t)
            else:
                log_image("license_image/clean",   img_clean)
                log_image("license_image/corrupt", img_corrupt_clone)
                log_image("license_image/encryption_license", reconstructions)
                # log_image("license_image/license", license)
                log_image("license_image/corrupt_encrypted", img_corrupt)
                log_image("license_image/recon",   img_recon)
                log_image("license_debug/pred_clean_traj", pred_x0s.reshape(-1, *xdim), nrow=len_t)
                log_image("license_debug/recon_traj",      xs.reshape(-1, *xdim),      nrow=len_t)
        else:
            log_image("image/clean",   img_clean)
            log_image("image/corrupt", img_corrupt)
            log_image("image/recon",   img_recon)
            log_image("debug/pred_clean_traj", pred_x0s.reshape(-1, *xdim), nrow=len_t)
            log_image("debug/recon_traj",      xs.reshape(-1, *xdim),      nrow=len_t)

        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()

    @torch.no_grad()
    def generation(self, opt, val_dataset, corrupt_method, license=None, corrupt_type=None):

        from torch.utils.data import DataLoader
        from tqdm import tqdm

        os.makedirs('/data/shixi/i2sb_licence/generation_result/'+opt.name, exist_ok=True)
        os.makedirs('/data/shixi/i2sb_licence/generation_result/'+opt.name+'/recon', exist_ok=True)
        os.makedirs('/data/shixi/i2sb_licence/generation_result/'+opt.name+'/clean', exist_ok=True)
        os.makedirs('/data/shixi/i2sb_licence/generation_result/'+opt.name+'/corrupt', exist_ok=True)
        os.makedirs('/data/shixi/i2sb_licence/generation_result/'+opt.name+'/corrupt_encrypted', exist_ok=True)
        profile_path = None
        if opt.profile_generation:
            profile_path = '/data/shixi/i2sb_licence/generation_result/'+opt.name+'/generation_profile.jsonl'
            if os.path.exists(profile_path):
                os.remove(profile_path)

        val_loader = DataLoader(val_dataset,
            batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,
        )

        for it, data in enumerate(val_loader, 0):
            if it >= 400:
                break

            img_clean, img_corrupt, mask, y, cond = self.sample_batch(opt, val_loader, corrupt_method, corrupt_type=corrupt_type, data=data)
            profile_records = []
            if license is not None:
                assert license in [0,1,2,3], "License type must be 0, 1, 2, or 3!"
                with torch.no_grad():
                    img_corrupt = img_corrupt.to(next(self.license_net.parameters()).device)
                    def _run_license_net():
                        encrypted = self.license_net(img_corrupt)
                        mixed = (1 - opt.gamma) * encrypted + opt.gamma * img_corrupt
                        return encrypted, mixed

                    if opt.profile_generation:
                        (img_corrupt_encrypted, x1), license_profile = self._profile_cuda_region(
                            "license_net",
                            [next(self.license_net.parameters()).device],
                            _run_license_net,
                            step="preprocess",
                        )
                        license_profile["batch_idx"] = it
                        profile_records.append(license_profile)
                    else:
                        img_corrupt_encrypted, x1 = _run_license_net()

            else:
                img_corrupt_encrypted = img_corrupt
                x1 = img_corrupt.to(opt.device)

            xs, pred_x0s = self.ddpm_sampling(
                opt,
                x1,
                mask=None,
                cond=cond,
                clip_denoise=opt.clip_denoise,
                verbose=opt.global_rank==0,
                profile_records=profile_records if opt.profile_generation else None,
            )
            if opt.profile_generation:
                for record in profile_records:
                    record["batch_idx"] = it
                self._append_profile_records(profile_path, profile_records)

            img_clean = img_clean.detach().cpu()
            img_corrupt = img_corrupt.detach().cpu()
            img_corrupt_encrypted = img_corrupt_encrypted.detach().cpu()
            xs = xs.detach().cpu()
            pred_x0s = pred_x0s.detach().cpu()

            batch, len_t, *xdim = xs.shape
            assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
            assert xs.shape == pred_x0s.shape

            img_recon = xs[:, 0, ...]
            
            img_recon = (img_recon + 1)/2
            img_clean = (img_clean + 1)/2
            img_corrupt = (img_corrupt + 1)/2
            img_corrupt_encrypted = (img_corrupt_encrypted + 1)/2
            if mask is not None and license is not None:
                mask = mask.cpu()
                img_recon = img_recon.cpu()
                img_corrupt = img_corrupt.cpu()
                img_recon = mask * img_recon + (1. - mask) * img_corrupt

                img_corrupt = (1. - mask) * img_clean + mask
            
            bs = img_recon.shape[0]
            for subimg in range(bs):
                tu.save_image(img_recon[subimg], '/data/shixi/i2sb_licence/generation_result/'+opt.name+'/recon/'+str(it*bs+subimg)+'.png')
                tu.save_image(img_clean[subimg], '/data/shixi/i2sb_licence/generation_result/'+opt.name+'/clean/'+str(it*bs+subimg)+'.png')
                tu.save_image(img_corrupt[subimg], '/data/shixi/i2sb_licence/generation_result/'+opt.name+'/corrupt/'+str(it*bs+subimg)+'.png')
                tu.save_image(img_corrupt_encrypted[subimg], '/data/shixi/i2sb_licence/generation_result/'+opt.name+'/corrupt_encrypted/'+str(it*bs+subimg)+'.png')

        torch.cuda.empty_cache()

    @torch.no_grad()
    def noencryption_generation(self, opt, val_dataset, corrupt_method, corrupt_type=None):

        from torch.utils.data import DataLoader

        os.makedirs('/data/shixi/i2sb_licence/generation_result/'+opt.name, exist_ok=True)
        os.makedirs('/data/shixi/i2sb_licence/generation_result/'+opt.name+'/recon', exist_ok=True)
        os.makedirs('/data/shixi/i2sb_licence/generation_result/'+opt.name+'/clean', exist_ok=True)
        os.makedirs('/data/shixi/i2sb_licence/generation_result/'+opt.name+'/corrupt', exist_ok=True)
        profile_path = None
        if opt.profile_generation:
            profile_path = '/data/shixi/i2sb_licence/generation_result/'+opt.name+'/generation_profile.jsonl'
            if os.path.exists(profile_path):
                os.remove(profile_path)

        val_loader = DataLoader(
            val_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=1,
            drop_last=False,
        )

        for it, data in enumerate(val_loader, 0):
            if it >= 400:
                break

            img_clean, img_corrupt, mask, y, cond = self.sample_batch(
                opt,
                val_loader,
                corrupt_method,
                corrupt_type=corrupt_type,
                data=data,
            )
            profile_records = []
            x1 = img_corrupt.to(opt.device)

            xs, pred_x0s = self.ddpm_sampling(
                opt,
                x1,
                mask=None,
                cond=cond,
                clip_denoise=opt.clip_denoise,
                verbose=opt.global_rank == 0,
                profile_records=profile_records if opt.profile_generation else None,
            )
            if opt.profile_generation:
                for record in profile_records:
                    record["batch_idx"] = it
                self._append_profile_records(profile_path, profile_records)

            img_clean = img_clean.detach().cpu()
            img_corrupt = img_corrupt.detach().cpu()
            xs = xs.detach().cpu()
            pred_x0s = pred_x0s.detach().cpu()

            batch, len_t, *xdim = xs.shape
            assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
            assert xs.shape == pred_x0s.shape

            img_recon = xs[:, 0, ...]
            img_recon = (img_recon + 1) / 2
            img_clean = (img_clean + 1) / 2
            img_corrupt = (img_corrupt + 1) / 2

            if mask is not None:
                mask = mask.cpu()
                img_recon = img_recon.cpu()
                img_corrupt = img_corrupt.cpu()
                img_recon = mask * img_recon + (1. - mask) * img_corrupt
                img_corrupt = (1. - mask) * img_clean + mask

            bs = img_recon.shape[0]
            for subimg in range(bs):
                index = it * bs + subimg
                tu.save_image(img_recon[subimg], f'/data/shixi/i2sb_licence/generation_result/{opt.name}/recon/{index}.png')
                tu.save_image(img_clean[subimg], f'/data/shixi/i2sb_licence/generation_result/{opt.name}/clean/{index}.png')
                tu.save_image(img_corrupt[subimg], f'/data/shixi/i2sb_licence/generation_result/{opt.name}/corrupt/{index}.png')

        torch.cuda.empty_cache()

    def train_dynamic_encryption_model(self, opt, train_dataset, val_dataset, corrupt_method):
        warning_image = self._load_warning_image(opt)

        self.writer = util.build_log_writer(opt)
        log = self.log

        net = self.encryption_model
        ema = self.encryption_model_ema
        # optimizer, sched = build_optimizer_sched(opt, net, log)

        from itertools import chain
        optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
        optimizer = AdamW(chain(net.parameters(), self.license_net.parameters()), **optim_dict)
        log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

        if opt.lr_gamma < 1.0:
            sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
            sched = lr_scheduler.StepLR(optimizer, **sched_dict)
            log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
        else:
            sched = None

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader   = util.setup_loader(val_dataset,   opt.microbatch)

        self.license_net = self.license_net.to("cuda:2")
        net.train()
        self.license_net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()
            
            # entropy_list = []
            # entropy_list2 = []

            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                x0, x1, mask, y, cond = self.sample_batch(opt, train_loader, corrupt_method, corrupt_type=opt.corrupt)

                x1 = x1.to("cuda:2")
                x1_license = self.license_net(x1)
                x1_license = x1_license * (1 - opt.gamma) + x1 * opt.gamma

                x1_license = x1_license.to("cuda:0")
                x1 = x1.to("cuda:0")

                losses = self._backward_single_outer_losses(
                    opt,
                    net,
                    x0,
                    x1,
                    cond,
                    warning_image,
                    x1_license,
                )

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss_auth:{} | loss_warning:{} | loss_bad:{} | loss_total:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(losses["auth"].item()),
                "{:+.4f}".format(losses["warning"].item()),
                "{:+.4f}".format(losses["unauthorized_clean"].item()),
                "{:+.4f}".format(losses["total"].item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss_auth', losses["auth"].detach())
                self.writer.add_scalar(it, 'loss_warning', losses["warning"].detach())
                self.writer.add_scalar(it, 'loss_bad', losses["unauthorized_clean"].detach())
                self.writer.add_scalar(it, 'loss_total', losses["total"].detach())

            if it % 50 == 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.encryption_model.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                    torch.save({
                        "net": self.license_net.state_dict(),
                    }, opt.ckpt_path / "license_latest.pt")
                    log.info(f"Saved latest({it=}) license checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()

            if it % 50 == 0:
                self.encryption_model.eval()
                self.license_net.eval()
                self.evaluation(opt, it, val_loader, corrupt_method)
                self.evaluation(opt, it, val_loader, corrupt_method, license=1, corrupt_type=opt.corrupt)
                self.encryption_model.train()
                self.license_net.train()
            torch.cuda.empty_cache()
        self.writer.close()