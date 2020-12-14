import os
import sys
import time

import cv2
import numpy as np
import argparse
import collections

import torch
from torch import nn
import torch.utils.data
from torch.nn import functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image, ImageDraw

from dataset import AlignCollate, RawDataset, LmdbDataset, LmdbDataset_2, RawDataset_2
from modules.feature_extraction import BasicBlock
from modules.sequence_modeling import BidirectionalLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_length = 416


class ResNet_upconv(nn.Module):

    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet_upconv, self).__init__()

        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16),
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[0],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[1],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(1, 2), padding=(1, 0))

        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[2],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)

        self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3],
                                 kernel_size=2, stride=(1, 2), padding=(1, 0), bias=False)

        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3],
                                 kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

        self.up_conv1 = nn.ConvTranspose2d(self.output_channel_block[3], self.output_channel_block[2],
                                           kernel_size=(2, 1), stride=(2, 1), bias=False)
        self.up_conv2 = nn.ConvTranspose2d(self.output_channel_block[2], self.output_channel_block[1],
                                           kernel_size=(2, 1), stride=(2, 1), bias=False)
        self.up_conv3 = nn.ConvTranspose2d(self.output_channel_block[1], self.output_channel_block[0],
                                           kernel_size=(2, 1), stride=(2, 1), bias=False)
        self.up_conv4 = nn.ConvTranspose2d(self.output_channel_block[0], 1,
                                           kernel_size=(2, 1), stride=(2, 1), bias=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # input - (batch_size, channel, H, W)
    def forward(self, x):
        x = self.conv0_1(x)  # (batch_size, 32, H, W)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)  # (batch_size, 64, H, W)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)  # (batch_size, 64, H/2, W/2)
        x = self.layer1(x)  # (batch_size, 128, H/2, W/2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)  # (batch_size, 128, H/4, W/4)
        x = self.layer2(x)  # (batch_size, 256, H/4, W/4)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)  # (batch_size, 256, H/8, W/4+1) or (batch_size, 256, H/4+1, W/8)
        x = self.layer3(x)  # (batch_size, 512, H/8, W/4+1) or (batch_size, 512, H/4+1, W/8)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)  # (batch_size, 512, H/16, W/4+2) or (batch_size, 512, H/4+2, W/16)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)  # (batch_size, 512, H/16-1, W/4+1) or (batch_size, 512, H/4+1, W/16-1)
        x = self.bn4_2(x)
        x = self.relu(x)

        x = self.up_conv1(x)
        x = self.relu(x)
        x = self.up_conv2(x)
        x = self.relu(x)
        x = self.up_conv3(x)
        x = self.relu(x)
        x = self.up_conv4(x)

        return x


class ResNet_nomaxpool(nn.Module):

    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet_nomaxpool, self).__init__()

        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16),
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[0],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[1],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(1, 2), padding=(1, 0))

        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[2],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)

        self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3],
                                 kernel_size=2, stride=(1, 2), padding=(1, 0), bias=False)

        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3],
                                 kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # input - (batch_size, channel, H, W)
    def forward(self, x):
        x = self.conv0_1(x)  # (batch_size, 32, H, W)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)  # (batch_size, 64, H, W)
        x = self.bn0_2(x)
        x = self.relu(x)

        # x = self.maxpool1(x)  # (batch_size, 64, H/2, W/2)
        x = self.layer1(x)  # (batch_size, 128, H/2, W/2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # x = self.maxpool2(x)  # (batch_size, 128, H/4, W/4)
        x = self.layer2(x)  # (batch_size, 256, H/4, W/4)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # x = self.maxpool3(x)  # (batch_size, 256, H/8, W/4+1) or (batch_size, 256, H/4+1, W/8)
        x = self.layer3(x)  # (batch_size, 512, H/8, W/4+1) or (batch_size, 512, H/4+1, W/8)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)  # (batch_size, 512, H/16, W/4+2) or (batch_size, 512, H/4+2, W/16)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)  # (batch_size, 512, H/16-1, W/4+1) or (batch_size, 512, H/4+1, W/16-1)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x


class ResNet_segment_text_upconv(nn.Module):
    def __init__(self, ckpt_path, freeze_CNN=True):
        super(ResNet_segment_text_upconv, self).__init__()
        self.output_dim = 512
        self.hidden_dim = 128
        self.CNN = ResNet_upconv(1, self.output_dim, BasicBlock, [1, 2, 5, 3])
        self.load_CNN_weight(ckpt_path)
        if freeze_CNN:
            self.CNN.requires_grad_(False)
        # self.Channel = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        '''
        self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.output_dim, self.hidden_dim, self.hidden_dim),
                BidirectionalLSTM(self.hidden_dim, self.hidden_dim, self.hidden_dim))
        '''
        # self.Prediction = nn.Linear(self.hidden_dim, 1)
        # self.Prediction = nn.Linear(self.output_dim, 1)
        # self.Loss = nn.SmoothL1Loss()
        # self.Loss = nn.BCELoss()

    def load_CNN_weight(self, ckpt_path):
        restore_ckpt = collections.OrderedDict()
        ckpt = torch.load(ckpt_path)
        for k, v in ckpt.items():
            if k.startswith('module.FeatureExtraction.ConvNet'):
                k = k.split('.')
                new_k = '.'.join(k[3:])
                restore_ckpt[new_k] = v
        self.CNN.load_state_dict(restore_ckpt, strict=False)

    def forward(self, img, logits=None, weights=None):
        visual_feature = self.CNN(img)

        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 2, 1, 3))
        visual_feature = visual_feature.squeeze(-1)

        # contextual_feature = self.SequenceModeling(visual_feature)
        contextual_feature = visual_feature

        # predicition = self.Prediction(contextual_feature.contiguous())
        predicition = contextual_feature
        predicition = predicition.squeeze(-1)
        predicition = torch.sigmoid(predicition)
        if logits is None:
            return predicition.detach()
        else:
            loss = nn.BCELoss(weight=weights)(predicition, logits)
            return loss


class ResNet_segment_text_bilstm(nn.Module):
    def __init__(self, ckpt_path, freeze_CNN=True):
        super(ResNet_segment_text_bilstm, self).__init__()
        self.output_dim = 512
        self.hidden_dim = 128
        self.CNN = ResNet_nomaxpool(1, self.output_dim, BasicBlock, [1, 2, 5, 3])
        self.load_CNN_weight(ckpt_path)
        if freeze_CNN:
            self.CNN.requires_grad_(False)
        # self.Channel = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.output_dim, self.hidden_dim, self.hidden_dim),
                BidirectionalLSTM(self.hidden_dim, self.hidden_dim, self.hidden_dim))

        self.Prediction = nn.Linear(self.hidden_dim, 1)
        # self.Prediction = nn.Linear(self.output_dim, 1)
        # self.Loss = nn.SmoothL1Loss()
        self.Loss = nn.BCELoss()

    def load_CNN_weight(self, ckpt_path):
        restore_ckpt = collections.OrderedDict()
        ckpt = torch.load(ckpt_path)
        for k, v in ckpt.items():
            if k.startswith('module.FeatureExtraction.ConvNet'):
                k = k.split('.')
                new_k = '.'.join(k[3:])
                restore_ckpt[new_k] = v
        self.CNN.load_state_dict(restore_ckpt, strict=False)

    def forward(self, img, logits=None, weights=None):
        visual_feature = self.CNN(img)

        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 2, 1, 3))
        visual_feature = visual_feature.squeeze(-1)

        contextual_feature = self.SequenceModeling(visual_feature)
        # contextual_feature = visual_feature

        predicition = self.Prediction(contextual_feature.contiguous())
        # predicition = contextual_feature
        predicition = predicition.squeeze(-1)
        predicition = torch.sigmoid(predicition)
        if logits is None:
            return predicition.detach()
        else:
            loss = self.loss(predicition, logits)
            return loss


class ResNet_segment_text_simple(nn.Module):
    def __init__(self, ckpt_path, freeze_CNN=True):
        super(ResNet_segment_text_simple, self).__init__()
        self.output_dim = 512
        self.hidden_dim = 128
        self.CNN = ResNet_nomaxpool(1, self.output_dim, BasicBlock, [1, 2, 5, 3])
        self.load_CNN_weight(ckpt_path)
        if freeze_CNN:
            self.CNN.requires_grad_(False)
        # self.Channel = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        # self.Prediction = nn.Linear(self.hidden_dim, 1)
        self.Prediction = nn.Linear(self.output_dim, 1)
        # self.Loss = nn.SmoothL1Loss()
        self.Loss = nn.BCELoss()

    def load_CNN_weight(self, ckpt_path):
        restore_ckpt = collections.OrderedDict()
        ckpt = torch.load(ckpt_path)
        for k, v in ckpt.items():
            if k.startswith('module.FeatureExtraction.ConvNet'):
                k = k.split('.')
                new_k = '.'.join(k[3:])
                restore_ckpt[new_k] = v
        self.CNN.load_state_dict(restore_ckpt, strict=False)

    def forward(self, img, logits=None, weights=None):
        visual_feature = self.CNN(img)

        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 2, 1, 3))
        visual_feature = visual_feature.squeeze(-1)

        # contextual_feature = self.SequenceModeling(visual_feature)
        contextual_feature = visual_feature

        predicition = self.Prediction(contextual_feature.contiguous())
        # predicition = contextual_feature
        predicition = predicition.squeeze(-1)
        predicition = torch.sigmoid(predicition)
        if logits is None:
            return predicition.detach()
        else:
            loss = self.loss(predicition, logits)
            return loss


def get_logits_and_weights(batch_size, labels, alpha, beta):
    logits = torch.ones((batch_size, feature_length))
    weights = torch.ones((batch_size, feature_length)) * alpha
    for i in range(batch_size):
        for label in labels[i]:
            label = int(label)
            logits[i][int((label - 1) * feature_length / 10000)] = 0
            weights[i][int((label - 1) * feature_length / 10000)] = beta
    return logits, weights


def nn_method_vertical_train(method='upconv', freeze_CNN=False):
    switch = {
        'upconv': ResNet_segment_text_upconv,
        'bilstm': ResNet_segment_text_bilstm,
        'simple': ResNet_segment_text_simple,
    }

    model = switch[method]('saved_models/Line_baseline_xl_2/best_accuracy.pth', freeze_CNN=freeze_CNN)
    model.to(device)
    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollater = AlignCollate(imgH=100, imgW=32, keep_ratio_with_pad=False)

    train_data = LmdbDataset_2(root='dataset/split_train')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4,
                                               collate_fn=AlignCollater, pin_memory=True)
    val_data = LmdbDataset_2(root='dataset/split_val')
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False, num_workers=4,
                                             collate_fn=AlignCollater, pin_memory=True)

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))

    optimizer = torch.optim.Adam(filtered_parameters, lr=3e-4)
    print("Optimizer:", optimizer)

    iteration = 0

    total_iter = 233333
    ckpt_step = 20000
    val_step = 20000
    output_step = 1000
    alpha = 0.9
    beta = 0.1

    while True:
        # train part
        for image, labels in train_loader:
            start_time = time.time()
            batch_size = image.size(0)
            image = image.to(device)
            labels = [label.split(',') for label in labels]
            logits, weights = get_logits_and_weights(batch_size, labels, alpha, beta)
            logits = logits.to(device)
            weights = weights.to(device)
            model.zero_grad()
            loss = model(image, logits, weights)
            loss.backward()
            optimizer.step()

            if (iteration + 1) % output_step == 0:
                print('[iter: {} / {}] loss: {}, time: {}'.format(
                    iteration + 1,
                    total_iter,
                    np.round(loss.item(), 4),
                    np.round(time.time() - start_time, 4)
                ), flush=True)

            if (iteration + 1) == total_iter:
                break

            # validation part
            if (iteration + 1) % val_step == 0 or iteration == 0:
                start_val_time = time.time()
                model.eval()
                with torch.no_grad():
                    # losses = []
                    gold = []
                    preds = []
                    for val_image, val_labels in val_loader:
                        val_batch_size = val_image.size(0)
                        val_image = val_image.to(device)
                        val_labels = [label.split(',') for label in val_labels]
                        val_logits, _ = get_logits_and_weights(val_batch_size, val_labels, alpha, beta)
                        val_logits = val_logits.to(device)
                        # val_loss = model(val_image, val_logits).item()
                        # losses.append(val_loss)
                        pred = model(val_image)
                        logits = val_logits.detach().cpu().int().numpy().reshape(-1).tolist()
                        pred = pred.detach().cpu().numpy()
                        pred = np.where(pred > 0.5, 1, 0).reshape(-1).tolist()
                        gold.extend(logits)
                        preds.extend(pred)
                    p = np.round(precision_score(gold, preds), 4)
                    r = np.round(recall_score(gold, preds), 4)
                    f1 = np.round(f1_score(gold, preds), 4)
                    print('[iter: {} / {}] p: {}, r: {}, f1: {}, time: {}'.format(
                        iteration + 1,
                        total_iter,
                        p, r, f1,
                        np.round(time.time() - start_val_time, 1)
                    ), flush=True)
                model.train()

            save_root = './saved_models/split_Upconv'

            # save model per 1e+5 iter.
            if (iteration + 1) % ckpt_step == 0:
                os.makedirs(save_root, exist_ok=True)
                torch.save(model.state_dict(), f'{save_root}/iter_{iteration + 1}.pth')

            if (iteration + 1) == total_iter:
                torch.save(model.state_dict(), f'{save_root}/iter_final_{iteration + 1}.pth')
                print('end the training')
                return

            iteration += 1


def nn_method_vertical(method, img_path, ckpt_path, score_threshold=1.0, NMS_threshold=20):
    switch = {
        'upconv': ResNet_segment_text_upconv,
        'bilstm': ResNet_segment_text_bilstm,
        'simple': ResNet_segment_text_simple,
    }
    model = switch[method]('saved_models/Line_baseline_xl_2/best_accuracy.pth')
    model.load_state_dict(torch.load(ckpt_path))
    model.to(device)
    val_dataset = RawDataset_2(root=img_path)
    AlignCollater = AlignCollate(imgH=100, imgW=32, keep_ratio_with_pad=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,
                                                 collate_fn=AlignCollater, pin_memory=True)
    data_lens = [4, 8, 17, 7, 7, 18, 16]
    for (val_image, val_image_path), data_len in zip(val_dataloader, data_lens):
        # batch_size = val_image.size(0)
        val_image = val_image.to(device)
        pred = model(val_image)
        pred = pred.detach().cpu().numpy().reshape(-1).tolist()
        candidate = [(score, idx) for idx, score in enumerate(pred)]
        if method == 'upconv':
            candidate.sort(reverse=False)
        else:
            candidate.sort(reverse=True)
        choosen_candidate = []
        if method == 'upconv':
            for score, idx in candidate:
                if score > score_threshold:
                    break
                NMS_flag = False
                for choosen_idx in choosen_candidate:
                    if abs(choosen_idx - idx) <= NMS_threshold:
                        NMS_flag = True
                        break
                if NMS_flag:
                    continue
                choosen_candidate.append(idx)
                if len(choosen_candidate) == data_len:
                    break
        else:
            for score, idx in candidate:
                if score < score_threshold:
                    break
                NMS_flag = False
                for choosen_idx in choosen_candidate:
                    if abs(choosen_idx - idx) <= NMS_threshold:
                        NMS_flag = True
                        break
                if NMS_flag:
                    continue
                choosen_candidate.append(idx)
        img = Image.open(val_image_path[0])
        draw = ImageDraw.Draw(img)
        width, height = img.size
        split_height = []
        for idx in choosen_candidate:
            draw_height = idx / feature_length * height
            split_height.append(draw_height)
            draw.line(((0, draw_height), (width - 1, draw_height)), fill=(255, 0, 0), width=2)
        img.save(os.path.join('result', os.path.basename(val_image_path[0])))


def cv_method_horizontal(img, threshold=2):
    """
    Segmentation of line image to single characters. Based on horizontal profile.
    """
    inverted = cv2.bitwise_not(img)
    inverted = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    print(inverted.shape)

    profile = inverted.sum(axis=0)
    print("Profile shape:", profile.shape)

    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    m = np.max(profile)
    print("Profile shape:", profile.shape)

    candidates = []
    state = 0  # 0 = gap
    lastGap = -1
    for i in range(profile.shape[0]):
        h = float(profile[i]) / m * 100
        if h <= threshold:  # gap
            # cv2.line(result, (i,0), (i, result.shape[0]), (0,255,0), 1)
            if state == 1:
                # print (lastGap, i)
                candidates.append((lastGap, i))
            lastGap = i
            state = 0

        else:
            state = 1

    for c in candidates:
        cv2.line(result, (c[0], 0), (c[0], result.shape[0]), (255, 0, 0), 1)
        cv2.line(result, (c[1], 0), (c[1], result.shape[0]), (255, 0, 0), 1)

    cv2.imshow('result', result)
    cv2.waitKey(0)

    return candidates


def cv_method_vertical(img, threshold=10):
    """
    Segmentation of line image to single characters. Based on vertical profile.
    """
    inverted = cv2.bitwise_not(img)
    _, inverted = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    profile = inverted.sum(axis=1)
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    m = np.max(profile)

    candidates = []
    state = 0  # 0 = gap
    lastGap = -1
    for i in range(profile.shape[0]):
        h = float(profile[i]) / m * 100
        if h <= threshold:  # gap
            # cv2.line(result, (i,0), (i, result.shape[0]), (0,255,0), 1)
            if state == 1:
                # print (lastGap, i)
                candidates.append((lastGap, i))
            lastGap = i
            state = 0

        else:
            state = 1

    for c in candidates:
        cv2.line(result, (0, c[0]), (result.shape[1], c[0]), (255, 0, 0), 1)
        cv2.line(result, (0, c[1]), (result.shape[1], c[1]), (0, 0, 255), 1)

    cv2.imwrite('result/cv2_segment.jpg', result)
    cv2.imshow('img', result)
    cv2.waitKey()

    return candidates


if __name__ == '__main__':
    # img = cv2.imread('test_line_image/true_line/20201024234424.png', 0)
    # cv_method_vertical(img)
    # nn_method_vertical_train(freeze_CNN=False)
    '''
    feature_length = 100
    nn_method_vertical('bilstm', 'test_line_image/true_line', 'saved_models/split/iter_200000.pth',
                       score_threshold=0.002)
    '''
    '''
    feature_length = 100
    nn_method_vertical('simple', 'test_line_image/true_line', 'saved_models/split_None/iter_200000.pth',
                       score_threshold=0.002)
    '''
    feature_length = 416
    nn_method_vertical('upconv', 'test_line_image/true_line', 'saved_models/split_Upconv/iter_220000.pth',
                       score_threshold=0.999)
