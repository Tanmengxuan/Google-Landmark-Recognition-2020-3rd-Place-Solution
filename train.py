import sys
sys.path.append('geffnet-20200820/')
sys.path.append('ResNeSt-master/')
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import math
import time
import pickle
import random
import argparse
import albumentations
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
from sklearn.metrics import cohen_kappa_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.backends import cudnn

#import apex
#from apex import amp
#from apex.parallel import DistributedDataParallel

from dataset import LandmarkDataset, get_df, get_transforms
from util import global_average_precision_score, GradualWarmupSchedulerV2
from models import DenseCrossEntropy, Swish_module
from models import ArcFaceLossAdaptiveMargin, Effnet_Landmark, RexNet20_Landmark, ResNest101_Landmark

use_cuda = torch.cuda.is_available()

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='/raid/GLD2')
    parser.add_argument('--train-step', type=int, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--init-lr', type=float, default=1e-4)
    parser.add_argument('--n-epochs', type=int, default=15)
    parser.add_argument('--start-from-epoch', type=int, default=1)
    parser.add_argument('--stop-at-epoch', type=int, default=999)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--load-from', type=str, default='')
    args, _ = parser.parse_known_args()
    return args


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer, criterion):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    #import pdb; pdb.set_trace()
    for (data, target) in bar:
        #import pdb; pdb.set_trace()
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        if not args.use_amp:
            logits_m = model(data)
            loss = criterion(logits_m, target)
            loss.backward()
            optimizer.step()
        else:
            logits_m = model(data)
            loss = criterion(logits_m, target)
            if use_cuda:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

        if use_cuda:
            torch.cuda.synchronize()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    return train_loss

def val_epoch(model, valid_loader, criterion, get_output=False):

    model.eval()
    val_loss = []
    PRODS_M = []
    PREDS_M = []
    TARGETS = []

    with torch.no_grad():
        for (data, target) in tqdm(valid_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            logits_m = model(data)

            lmax_m = logits_m.max(1)
            probs_m = lmax_m.values
            preds_m = lmax_m.indices

            PRODS_M.append(probs_m.detach().cpu())
            PREDS_M.append(preds_m.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits_m, target)
            val_loss.append(loss.detach().cpu().numpy())

        val_loss = np.mean(val_loss)
        PRODS_M = torch.cat(PRODS_M).numpy()
        PREDS_M = torch.cat(PREDS_M).numpy()
        TARGETS = torch.cat(TARGETS)

    if get_output:
        return LOGITS_M
    else:
        acc_m = (PREDS_M == TARGETS.numpy()).mean() * 100.
        y_true = {idx: target if target >=0 else None for idx, target in enumerate(TARGETS)}
        y_pred_m = {idx: (pred_cls, conf) for idx, (pred_cls, conf) in enumerate(zip(PREDS_M, PRODS_M))}
        gap_m = global_average_precision_score(y_true, y_pred_m)
        return val_loss, acc_m, gap_m


def main():

    # get dataframe
    df, out_dim = get_df(args.kernel_type, args.data_dir, args.train_step)
    print(f"out_dim = {out_dim}")

    #df = df.sample(frac=0.8, random_state=66).reset_index(drop=True)
    # get adaptive margin
    tmp = np.sqrt(1 / np.sqrt(df['landmark_id'].value_counts().sort_index().values))
    margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05

    # get augmentations
    transforms_train, transforms_val = get_transforms(args.image_size)

    # get train and valid dataset
    df_train = df[df['fold'] != args.fold]
    df_valid = df[df['fold'] == args.fold].reset_index(drop=True).query("index % 15==0")

    dataset_train = LandmarkDataset(df_train, 'train', 'train', transform=transforms_train)
    dataset_valid = LandmarkDataset(df_valid, 'train', 'val', transform=transforms_val)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)

    # model
    #import pdb; pdb.set_trace()
    model = ModelClass(args.enet_type, out_dim=out_dim)
    if use_cuda:
        model = model.cuda()
        #model = apex.parallel.convert_syncbn_model(model)

    # loss func
    def criterion(logits_m, target):
        arc = ArcFaceLossAdaptiveMargin(margins=margins, s=80)
        loss_m = arc(logits_m, target, out_dim)
        return loss_m

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    if args.use_amp:
        if use_cuda:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # load pretrained
    if len(args.load_from) > 0:
        if use_cuda:
            checkpoint = torch.load(args.load_from,  map_location='cuda:{}'.format(args.local_rank))
        else:
            checkpoint = torch.load(args.load_from)
        state_dict = checkpoint['model_state_dict']
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
        if args.train_step==1:
            del state_dict['metric_classify.weight']
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=True)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
               args.start_from_epoch = checkpoint['epoch'] + 1
            if 'gap_m_best' in checkpoint:
                checkpoint_gap_m_best = checkpoint['gap_m_best']
            else:
                checkpoint_gap_m_best = 0.3476 #temp
        del checkpoint, state_dict
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    #if use_cuda:
    #    model = DistributedDataParallel(model, delay_allreduce=True)

    # lr scheduler
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs-1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    # train & valid loop
    gap_m_max = 0.
    model_file = os.path.join(args.model_dir, f'{args.kernel_type}_fold{args.fold}.pth')
    gap_m_best = -1
    if len(args.load_from) > 0:
        gap_m_best = checkpoint_gap_m_best
        print(f" \n Load gap_m_best: {gap_m_best}\n")

    for epoch in range(args.start_from_epoch, args.n_epochs+1):

        print(time.ctime(), 'Epoch:', epoch)
        if len(args.load_from) > 0 and epoch == args.start_from_epoch:
            scheduler_warmup.step(epoch - 1)
            scheduler_warmup.step(epoch - 1) # Hack to control scheduler
        else:
            scheduler_warmup.step(epoch - 1)
        print(f"\nLearning RATE before train: {optimizer.param_groups[0]['lr']}\n")
        print(f"\nstep_Count: {optimizer._step_count}\n")
        #if use_cuda:
        #    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        #    train_sampler.set_epoch(epoch)
        #else:
        #    train_sampler = None
        train_sampler = None

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                                  shuffle=train_sampler is None, sampler=train_sampler, drop_last=True)

        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, acc_m, gap_m = val_epoch(model, valid_loader, criterion)

        if args.local_rank == 0:
            content = time.ctime() + ' ' + f'Fold {args.fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, valid loss: {(val_loss):.5f}, acc_m: {(acc_m):.6f}, gap_m: {(gap_m):.6f}.'
            print(f'\n{content}\n')
            with open(os.path.join(args.log_dir, f'{args.kernel_type}.txt'), 'a') as appender:
                appender.write(content + '\n')

            if gap_m > gap_m_best:
                gap_m_best = gap_m
                print('\n Saving Best Model...\n')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'gap_m': gap_m,
                    'gap_m_best': gap_m_best,
                }, os.path.join(args.model_dir, f'{args.kernel_type}_fold{args.fold}_best.pth'))

            print('gap_m_max ({:.6f} --> {:.6f}). Saving model ...'.format(gap_m_max, gap_m))
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'gap_m': gap_m,
                    'gap_m_best': gap_m_best,
                    }, model_file)
            gap_m_max = gap_m

        if epoch == args.stop_at_epoch:
            print(time.ctime(), 'Training Finished!')
            break

    #torch.save({
    #    'epoch': epoch,
    #    'model_state_dict': model.state_dict(),
    #    'optimizer_state_dict': optimizer.state_dict(),
    #}, os.path.join(args.model_dir, f'{args.kernel_type}_fold{args.fold}_final.pth'))


if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    if args.enet_type == 'nest101':
        ModelClass = ResNest101_Landmark
    elif args.enet_type == 'rex20':
        ModelClass = RexNet20_Landmark
    else:
        ModelClass = Effnet_Landmark

    set_seed(0)

    if args.CUDA_VISIBLE_DEVICES != '-1':
        torch.backends.cudnn.benchmark = True
        if use_cuda:
            torch.cuda.set_device(args.local_rank)
            #torch.distributed.init_process_group(backend='nccl', init_method='env://')
        cudnn.benchmark = True

    main()
