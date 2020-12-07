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

from dataset import AlignCollate, RawDataset, LmdbDataset
from modules.ResNet_Shallow import ResNet


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet_segment_text(nn.Module):
    def __init__(self, ckpt_path, free_CNN=True):
        super(ResNet_segment_text, self).__init__()

        self.CNN = ResNet(1, 512, BasicBlock, [1, 2, 5, 3], 'vertical')

        self.CNN.requires_grad_(False)

        self.load_CNN_weight(ckpt_path)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
        self.Prediction = nn.Linear(512, 2)
        self.Loss = nn.BCELoss()

    def load_CNN_weight(self, ckpt_path):
        restore_ckpt = collections.OrderedDict()
        ckpt = torch.load(ckpt_path)
        for k, v in ckpt.items():
            if k.startswith('module.FeatureExtraction.ConvNet'):
                k = k.split('.')
                new_k = '.'.join(k[3:])
                restore_ckpt[new_k] = v
        self.CNN.load_state_dict(restore_ckpt)

    def forward(self, img, logits=None):
        visual_feature1, visual_feature2 = self.CNN(img)
        visual_feature = visual_feature1

        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 2, 1, 3))
        visual_feature = visual_feature.squeeze(3)

        prediciton = self.predict(visual_feature.contiguous())
        if logits is None:
            return prediciton.detach()
        else:
            loss = self.Loss(prediciton, logits)
            return loss


def nn_method_vertical_train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet_segment_text('saved_models/Line_baseline_xl/best_norm_ED.pth')
    model.to(device)
    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=100, imgW=32, keep_ratio_with_pad=False)
    Opt = collections.namedtuple('Opt', ['data_filtering_off'])
    opt = Opt(True)

    train_data = LmdbDataset(root='datasets/split', opt=opt)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False, num_workers=4,
                                              collate_fn=AlignCollate_demo, pin_memory=True)
    val_data = LmdbDataset(root='datasets/split', opt=opt)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False, num_workers=4,
                                             collate_fn=AlignCollate_demo, pin_memory=True)

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))

    optimizer = torch.optim.Adam(filtered_parameters, lr=3e-4)
    print("Optimizer:", optimizer)

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = 0

    feature_length = 100

    while True:
        # train part
        for image_tensors, labels in train_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            labels = [','.split(label) for label in labels]
            logits = torch.zeros((batch_size, feature_length))
            for i in range(batch_size):
                for label in labels:
                    label = int(label)
                    logits[label / feature_length] = 1
            logits = logits.to(device)
            model.zero_grad()
            loss = model(image, logits)
            loss.backward()
            optimizer.step()

            iteration += 1

            if (iteration + 1) == 100000:
                break

        # validation part
        if (iteration + 1) % 20000 == 0 or iteration == 0:  # To see training progress, we also conduct validation when 'iteration == 0'

            # for log
            with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a', encoding='utf-8') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, opt)
                model.train()

                # training loss and validation loss
                loss_log = f'[{iteration + 1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.4f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.4f}, {"Best_norm_ED":17s}: {best_norm_ED:0.4f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # save model per 1e+5 iter.
        if (iteration + 1) % 20000 == 0:
            os.makedirs('./saved_models/split/', exist_ok=True)
            torch.save(model.state_dict(), f'./saved_models/split/iter_{iteration + 1}.pth')

        if (iteration + 1) == 100000:
            torch.save(model.state_dict(), f'./saved_models/split/iter_{iteration + 1}.pth')
            print('end the training')
            sys.exit()


def nn_method_vertical(img):
    net = ResNet_segment_text('saved_models/Line_baseline_xl/best_norm_ED.pth')
    net.load_state_dict(torch.load('saved_models/Segment/best.pth'))


def cv_method_horizontal(img):
    '''
    Segmentation of line image to single characters. Based on horizontal profile.
    '''
    inverted = cv2.bitwise_not(img)
    inverted = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    print(inverted.shape)

    profile = inverted.sum(axis=0)
    print("Profile shape:", profile.shape)

    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    m = np.max(profile)
    print("Profile shape:", profile.shape)

    candidates = []
    threshold = 2
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


def cv_method_vertical(img):
    '''
    Segmentation of line image to single characters. Based on horizontal profile.
    '''
    inverted = cv2.bitwise_not(img)
    inverted = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    profile = inverted.sum(axis=1)
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    m = np.max(profile)

    candidates = []
    threshold = 10
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

    return candidates


if __name__ == '__main__':
    img = cv2.imread('test_line_image/true_line/20201024234320.png', 0)
    # cv_method_vertical(img)
    nn_method_vertical(img)