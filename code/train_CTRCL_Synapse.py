# -*- coding: utf-8 -*-
# Author: Xiangde Luo
# Date:   16 Dec. 2021
# Implementation for Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer.
# # Reference:
#   @article{luo2021ctbct,
#   title={Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer},
#   author={Luo, Xiangde and Hu, Minhao and Song, Tao and Wang, Guotai and Zhang, Shaoting},
#   journal={arXiv preprint arXiv:2112.04894},
#   year={2021}}
#   In the original paper, we don't use the validation set to select checkpoints and use the last iteration to inference for all methods.
#   In addition, we combine the validation set and test set to report the results.
#   We found that the random data split has some bias (the validation set is very tough and the test set is very easy).
#   Actually, this setting is also a fair comparison.
#   download pre-trained model to "code/pretrained_ckpt" folder, link:https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY

import argparse
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import get_config
from dataloaders import utils
from dataloaders.dataset_synapse import (Synapse_dataset, RandomGenerator)
from networks.net_factory import net_factory

from networks.mit_b2_segformer import MiT

from networks.resnet50_fpn import ResNet

from utilsss import losses, metrics, ramps
from val_2D import test_single_volume


parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str,
                    default='Synapse/Cross_Teaching_Between_CNN_Transformer', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=55400, help='maximum epoch number to train')  # 200 epochs
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=3e-4,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1234, help='random seed')
parser.add_argument('--num_classes', type=int,  default=9,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
# costs
# parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
# parser.add_argument('--consistency_type', type=str,
#                     default="mse", help='consistency_type')
# parser.add_argument('--consistency', type=float,
#                     default=1, help='consistency')
# parser.add_argument('--consistency_rampup', type=float,
#                     default=200.0, help='consistency_rampup')
args = parser.parse_args()
config = get_config(args)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def one_hot_encoder(n_classes, input_tensor):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


def find_same_elements(tensor1, tensor2):
    if tensor1.size() != tensor2.size():
        raise ValueError("Shape Error!")

    same_mask = torch.eq(tensor1, tensor2).int()

    return same_mask


def category_perception_map(feature, label, num_class):
    mask = one_hot_encoder(num_class, label)
    b_m, c_m, h_m, w_m = mask.shape
    prototype_list = []
    prototype_mean_list = []
    distance_list = []
    cpm_list = []

    for c in range(c_m):
        for b in range(b_m):
            if torch.sum(mask[b, c, :, :]) == 0:
                nn.Identity()
            else:
                prototype = torch.sum(feature[b, :, :, :].unsqueeze(0) * mask[b, c, :, :].unsqueeze(0).unsqueeze(1), dim=(2, 3), keepdim=True) / torch.sum(mask[b, c, :, :].unsqueeze(0).unsqueeze(1), dim=(2, 3), keepdim=True)
                prototype_list.append(prototype)

        if len(prototype_list) == 0:
            prototype_mean = 0
        else:
            prototype_mean = torch.mean(torch.cat(prototype_list, dim=0), dim=0, keepdim=True)
            prototype_list.clear()

        prototype_mean_list.append(prototype_mean)

    for b in range(b_m):
        for c in range(c_m):
            if torch.sum(mask[b, c, :, :]) == 0:
                nn.Identity()
            else:
                distance = 1 - F.cosine_similarity(feature[b, :, :, :].unsqueeze(0), prototype_mean_list[c], dim=1).unsqueeze(1)
                distance_list.append(distance)

        distance_cat = torch.cat(distance_list, dim=1)
        distance_list.clear()
        cpm = torch.softmax(-20 * distance_cat, dim=1)
        cpm_list.append(cpm)

    out = torch.cat(cpm_list, dim=1)
    cpm_list.clear()

    return out


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    c_model = ResNet(num_class=num_classes).cuda()
    t_model = MiT(num_class=num_classes).cuda()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = Synapse_dataset(root='../dataset/Synapse/train_npz/', mode="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = Synapse_dataset(root='../dataset/Synapse/test_vol_h5/', mode="val", transform=None)

    total_slices = len(db_train)
    print("Total slices is: {}".format(total_slices))

    trainloader = DataLoader(db_train, batch_size=args.batch_size, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    c_model.train()
    t_model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer1 = optim.AdamW(c_model.parameters(), lr=base_lr, weight_decay=0.0001)
    optimizer2 = optim.AdamW(t_model.parameters(), lr=base_lr, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    c_best_performance = 0.0
    t_best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # simultaneously
            c_outputs, c_feature_encoder, c_feature_decoder = c_model(volume_batch)
            c_outputs_soft = torch.softmax(c_outputs, dim=1)

            t_outputs, t_feature_encoder, t_feature_decoder = t_model(volume_batch)
            t_outputs_soft = torch.softmax(t_outputs, dim=1)

            # pixel-wise segmentation loss
            c_loss_seg = F.cross_entropy(c_outputs, label_batch.long())
            t_loss_seg = F.cross_entropy(t_outputs, label_batch.long())

            # rectified logit-wise collaborative learning
            c_pseudo = torch.argmax(c_outputs_soft.detach(), dim=1, keepdim=False)
            t_pseudo = torch.argmax(t_outputs_soft.detach(), dim=1, keepdim=False)

            c_true = find_same_elements(c_pseudo, label_batch)
            c_false = 1 - c_true
            t_true = find_same_elements(t_pseudo, label_batch)
            t_false = 1 - t_true

            label_batch_one_hot = one_hot_encoder(num_classes, label_batch.unsqueeze(1))

            c_lambda_a = 1 / (1 + torch.max(c_outputs_soft, dim=1)[0] - torch.sum(c_outputs_soft * label_batch_one_hot, dim=1))
            t_lambda_a = 1 / (1 + torch.max(t_outputs_soft, dim=1)[0] - torch.sum(t_outputs_soft * label_batch_one_hot, dim=1))

            c_lambda_s = torch.sum(c_outputs_soft * label_batch_one_hot, dim=1)
            t_lambda_s = torch.sum(t_outputs_soft * label_batch_one_hot, dim=1)

            c_lambda_c = torch.exp(torch.sum(c_outputs_soft * torch.log(c_outputs_soft + 1e-16), dim=1))
            t_lambda_c = torch.exp(torch.sum(t_outputs_soft * torch.log(t_outputs_soft + 1e-16), dim=1))

            c_lambda = (c_lambda_a * c_lambda_s * c_lambda_c).unsqueeze(1)
            t_lambda = (t_lambda_a * t_lambda_s * t_lambda_c).unsqueeze(1)

            c_outputs_soft_calibrated = c_false.unsqueeze(1) * (c_lambda * c_outputs_soft + (1 - c_lambda) * label_batch_one_hot) + c_true.unsqueeze(1) * c_outputs_soft
            t_outputs_soft_calibrated = t_false.unsqueeze(1) * (t_lambda * t_outputs_soft + (1 - t_lambda) * label_batch_one_hot) + t_true.unsqueeze(1) * t_outputs_soft

            c_loss_rlcl = torch.mean(torch.sum(F.kl_div(torch.log(c_outputs_soft), t_outputs_soft_calibrated.detach(), reduction='none'), dim=1))
            t_loss_rlcl = torch.mean(torch.sum(F.kl_div(torch.log(t_outputs_soft), c_outputs_soft_calibrated.detach(), reduction='none'), dim=1))

            # class-aware feature-wise collaborative learning
            # encoder
            label_batch_encoder = F.interpolate(label_batch.unsqueeze(1).float(), scale_factor=0.03125, mode='nearest')
            c_cpm_encoder = category_perception_map(c_feature_encoder, label_batch_encoder, 9)
            t_cpm_encoder = category_perception_map(t_feature_encoder, label_batch_encoder, 9)

            c_loss_cfcl_encoder = torch.mean(torch.sum(F.kl_div(torch.log(c_cpm_encoder), t_cpm_encoder.detach(), reduction='none'), dim=1) / batch_size)
            t_loss_cfcl_encoder = torch.mean(torch.sum(F.kl_div(torch.log(t_cpm_encoder), c_cpm_encoder.detach(), reduction='none'), dim=1) / batch_size)

            # decoder
            label_batch_decoder = F.interpolate(label_batch.unsqueeze(1).float(), scale_factor=0.25, mode='nearest')
            c_cpm_decoder = category_perception_map(c_feature_decoder, label_batch_decoder, 9)
            t_cpm_decoder = category_perception_map(t_feature_decoder, label_batch_decoder, 9)

            c_loss_cfcl_decoder = torch.mean(torch.sum(F.kl_div(torch.log(c_cpm_decoder), t_cpm_decoder.detach(), reduction='none'), dim=1) / batch_size)
            t_loss_cfcl_decoder = torch.mean(torch.sum(F.kl_div(torch.log(t_cpm_decoder), c_cpm_decoder.detach(), reduction='none'), dim=1) / batch_size)

            c_loss = c_loss_seg + 3 * c_loss_rlcl + c_loss_cfcl_encoder + 2 * c_loss_cfcl_decoder
            t_loss = t_loss_seg + 3 * t_loss_rlcl + t_loss_cfcl_encoder + 2 * t_loss_cfcl_decoder

            optimizer1.zero_grad()
            c_loss.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            t_loss.backward()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            # writer.add_scalar(
            #     'mkd_weight/mkd_weight', mkd_weight, iter_num)
            writer.add_scalar('loss/c_loss',
                              c_loss, iter_num)
            writer.add_scalar('loss/t_loss',
                              t_loss, iter_num)
            writer.add_scalar('loss/c_loss_seg',
                              c_loss_seg, iter_num)
            writer.add_scalar('loss/t_loss_seg',
                              t_loss_seg, iter_num)
            writer.add_scalar('loss/c_loss_rlcl',
                              c_loss_rlcl, iter_num)
            writer.add_scalar('loss/t_loss_rlcl',
                              t_loss_rlcl, iter_num)
            writer.add_scalar('loss/c_loss_cfcl_encoder',
                              c_loss_cfcl_encoder, iter_num)
            writer.add_scalar('loss/t_loss_cfcl_encoder',
                              t_loss_cfcl_encoder, iter_num)
            writer.add_scalar('loss/c_loss_cfcl_decoder',
                              c_loss_cfcl_decoder, iter_num)
            writer.add_scalar('loss/t_loss_cfcl_decoder',
                              t_loss_cfcl_decoder, iter_num)
            logging.info('iteration %d : c_model loss : %f t_model loss : %f' % (
                iter_num, c_loss.item(), t_loss.item()))
            if iter_num % 50 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    c_outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/c_model_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    t_outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/t_model_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 277 == 0:
                c_model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], c_model, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/c_model_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i], iter_num)
                    # writer.add_scalar('info/c_model_val_{}_hd95'.format(class_i+1),
                    #                   metric_list[class_i, 1], iter_num)

                c_performance = np.mean(metric_list, axis=0)

                # c_mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/c_model_val_mean_dice',
                                  c_performance, iter_num)
                # writer.add_scalar('info/c_model_val_mean_hd95',
                #                   c_mean_hd95, iter_num)

                save_mode_path = os.path.join(snapshot_path,
                                              'c_model_iter_{}_dice_{}.pth'.format(iter_num, round(c_performance, 4)))
                torch.save(c_model.state_dict(), save_mode_path)

                # if c_performance > c_best_performance:
                #     c_best_performance = c_performance
                #     save_mode_path = os.path.join(snapshot_path,
                #                                   'c_model_iter_{}_dice_{}.pth'.format(
                #                                       iter_num, round(c_best_performance, 4)))
                #     save_best = os.path.join(snapshot_path,
                #                              '{}_best_c_model.pth'.format(args.model))
                #     torch.save(c_model.state_dict(), save_mode_path)
                #     torch.save(c_model.state_dict(), save_best)

                logging.info(
                    'iteration %d : c_model_mean_dice : %f' % (iter_num, c_performance))
                c_model.train()

                t_model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], t_model, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/t_model_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i], iter_num)
                    # writer.add_scalar('info/t_model_val_{}_hd95'.format(class_i+1),
                    #                   metric_list[class_i, 1], iter_num)

                t_performance = np.mean(metric_list, axis=0)

                # t_mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/t_model_val_mean_dice',
                                  t_performance, iter_num)
                # writer.add_scalar('info/t_model_val_mean_hd95',
                #                   t_mean_hd95, iter_num)

                save_mode_path = os.path.join(snapshot_path,
                                              't_model_iter_{}_dice_{}.pth'.format(iter_num, round(t_performance, 4)))
                torch.save(t_model.state_dict(), save_mode_path)

                # if t_performance > t_best_performance:
                #     t_best_performance = t_performance
                #     save_mode_path = os.path.join(snapshot_path,
                #                                   't_model_iter_{}_dice_{}.pth'.format(
                #                                       iter_num, round(t_best_performance, 4)))
                #     save_best = os.path.join(snapshot_path,
                #                              '{}_best_t_model.pth'.format(args.model))
                #     torch.save(t_model.state_dict(), save_mode_path)
                #     torch.save(t_model.state_dict(), save_best)

                logging.info(
                    'iteration %d : t_model_mean_dice : %f' % (iter_num, t_performance))
                t_model.train()

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}/{}_RCL_checkpoints".format(
        args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
