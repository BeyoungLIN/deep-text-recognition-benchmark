
import os
import sys

import argparse
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from bs4 import BeautifulSoup, NavigableString
import editdistance
import random

import torch
from torch import nn
from torch.backends import cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import AlignCollate, PILDataset
from model import Model
from utils import CTCLabelConverter, AttnLabelConverter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_EXT = {'.jpg', '.tif', '.tiff', '.png'}


def get_already_done(done_path):
    done = set()
    if not os.path.isfile(done_path):
        return done
    with open(done_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            done.add(line.rstrip())
    return done


def get_count(count_path):
    if not os.path.isfile(count_path):
        return 0
    else:
        return int(open(count_path, 'r', encoding='utf-8').readline().rstrip())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--done_path', type=str, default='result/done.txt')
    parser.add_argument('--count_path', type=str, default='result/count.txt')
    parser.add_argument('--todo_file', type=str, default=None)
    parser.add_argument('--type', type=str, required=True, choices=['dingxiu', 'diaolong'])
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--strict', action='store_true')

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    parser.add_argument('--page_orient', type=str, choices=['horizontal', 'vertical', 'single'], default='horizontal',
                        help='page orientation, or single char')

    args = parser.parse_args()
    return args


def get_box(db_path):
    boxes = list()
    with open(db_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            line = line.strip().split(',')
            if len(line) % 2 == 1:
                line = line[:-1]
            line = list(map(int, line))
            l = min(line[::2])
            r = max(line[::2])
            u = min(line[1::2])
            d = max(line[1::2])
            boxes.append((l, u, r, d))
    return boxes


def get_gt(html_path, type):
    gt = []
    if type == 'dingxiu':
        with open(html_path, 'r', encoding='utf-8') as fp:
            html_file = fp.read()
        soup = BeautifulSoup(html_file, 'lxml')
        lines = soup.find_all('div', {'class': 'linespan'})
        for line in lines:
            contents = line.contents
            cur_line = ''
            br_cnt = 0
            for content in contents:
                if isinstance(content, NavigableString):
                    cur_line += content
                    br_cnt = 0
                else:
                    br_cnt += 1
                    if br_cnt > 1 and len(cur_line) > 0:
                        gt.append(cur_line)
                        cur_line = ''
            if len(cur_line) > 0:
                gt.append(cur_line)
                cur_line = ''
    elif type == 'diaolong':
        pass
    else:
        raise ValueError
    return gt


def demo(args, PIL_image_list, model, AlignCollate_demo, converter):
    dataset = PILDataset(args, PIL_image_list)
    demo_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=int(args.workers), collate_fn=AlignCollate_demo,
                             pin_memory=True)
    total_preds_str = []
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([args.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, args.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in args.Prediction:
                preds = model(image, text_for_pred)
                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)
            else:
                preds, alphas = model(image, text_for_pred, is_train=False)
                _, preds_index = preds.max(dim=2)
                preds_str = converter.decode(preds_index, length_for_pred)

            for pred in preds_str:
                if 'Attn' in args.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                total_preds_str.append(pred)
    return total_preds_str


def check_relax(gt, pred):
    if abs(len(gt) - len(pred)) >= 2:
        return False
    dis = editdistance.distance(gt, pred)
    if dis < max(len(gt), len(pred)) * 0.5:
        return True


def check_strict(gt, pred):
    if abs(len(gt) - len(pred)) >= 2:
        return False
    dis = editdistance.distance(gt, pred)
    if dis < max(len(gt), len(pred)) * 0.3:
        return True


def get_match_idx(preds, gts, strict=False):
    if strict:
        check = check_strict
    else:
        check = check_relax
    matching_idxs = []
    preds_len = len(preds)
    gts_len = len(gts)
    current_pred_idx = 0
    for current_gt_idx in range(gts_len):
        flag = 0
        current_gt = gts[current_gt_idx]
        while True:
            current_pred = preds[current_pred_idx]
            if check(current_gt, current_pred):
                matching_idxs.append((current_pred_idx, current_gt_idx))
                break
            else:
                current_pred_idx += 1
                flag += 1
            # max forward match distance = 7
            # if match fail, go back 3
            if flag == 7 or current_pred_idx == preds_len:
                current_pred_idx -= 10
                current_pred_idx = max(0, current_pred_idx)
                break
    return matching_idxs


def get_match_img(args, model, AlignCollate_demo, converter,
                  img_path, db_path, html_path, current_cnt):
    match_cnt = 0
    boxes = get_box(db_path)
    gts = get_gt(html_path, args.type)
    PIL_image_list = list()
    img = Image.open(img_path).convert('L')
    for box in boxes:
        crop_img = img.crop(box)
        # crop_img.show()
        PIL_image_list.append(crop_img)
    pred = demo(args, PIL_image_list, model, AlignCollate_demo, converter)
    matching_idx = get_match_idx(pred, gts, args.strict)
    for p_idx, g_idx in matching_idx:
        img = PIL_image_list[p_idx]
        img_gt = gts[g_idx]
        img_save_path = os.path.join(args.output_path, 'imgs', str(current_cnt) + '.jpg')
        img.save(img_save_path)
        gt_save_path = os.path.join(args.output_path, 'gts', str(current_cnt) + '.txt')
        with open(gt_save_path, 'w', encoding='utf-8') as fp:
            fp.write(img_gt + '\n')
        current_cnt += 1
        match_cnt += 1
    return match_cnt, current_cnt


def init_model(args):
    """ model configuration """
    if 'CTC' in args.Prediction:
        converter = CTCLabelConverter(args.character)
    else:
        converter = AttnLabelConverter(args.character)
    args.num_class = len(converter.character)

    if args.rgb:
        args.input_channel = 3
    model = Model(args)
    print('model input parameters', args.imgH, args.imgW, args.num_fiducial, args.input_channel, args.output_channel,
          args.hidden_size, args.num_class, args.batch_max_length, args.Transformation, args.FeatureExtraction,
          args.SequenceModeling, args.Prediction)
    try:
        model = torch.nn.DataParallel(model).to(device)
    except RuntimeError:
        raise RuntimeError(device)

    # load model
    print('loading pretrained model from %s' % args.saved_model)
    model.load_state_dict(torch.load(args.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=args.imgH, imgW=args.imgW, keep_ratio_with_pad=args.PAD)
    model.eval()
    return converter, model, AlignCollate_demo


if __name__ == '__main__':
    args = parse_args()
    done = get_already_done(args.done_path)
    print('{} imgs already done.'.format(len(done)))
    cnt = get_count(args.count_path)
    print('{} lines has been extracted.'.format(cnt))
    if args.todo_file is None:
        print('no todo_file in args. Scan input_path.')
        todo_list = list()
        for root, dirs, files in os.walk(args.input_path):
            for file in files:
                if os.path.splitext(file)[-1].lower() not in IMG_EXT:
                    continue
                file_path = os.path.join(root, file)
                if file_path in done:
                    continue
                todo_list.append(file_path)
    else:
        print('load todo_file from {}.'.format(args.todo_file))
        todo_list = [s.rstrip() for s in open(args.todo_file, 'r', encoding='utf-8').readlines()]
    print('{} files to do.'.format(len(todo_list)), flush=True)
    if args.shuffle:
        print('shuffle todo files.')
        random.shuffle(todo_list)
    os.makedirs(os.path.join(args.output_path, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'gts'), exist_ok=True)

    if args.character in ['CN-s', 'CN-m', 'CN-l', 'CN-xl']:
        size = args.character.split('-')[-1]
        with open('charset/charset_' + size + '.txt', 'r', encoding='utf-8') as chars:
            charset = [c.strip() for c in chars]
        charset = ''.join(charset)
        args.character = charset
    elif args.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        args.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    else:
        raise ValueError

    cudnn.benchmark = True
    cudnn.deterministic = True
    args.num_gpu = torch.cuda.device_count()

    converter, model, AlignCollate_demo = init_model(args)

    for file_path in todo_list:
        db_result_path = os.path.splitext(file_path)[0] + '.txt'
        if not os.path.isfile(db_result_path):
            continue
        html_gt_path = os.path.splitext(file_path)[0] + '.html'
        print('started from No. {} '.format(cnt), end='')
        match_cnt, cnt = get_match_img(args, model, AlignCollate_demo, converter,
                                       file_path, db_result_path, html_gt_path, cnt)
        print('{} get {} match imgs'.format(file_path, match_cnt))
        with open(args.done_path, 'a', encoding='utf-8') as fp:
            fp.write(file_path + '\n')
        with open(args.count_path, 'w', encoding='utf-8') as fp:
            fp.write(str(cnt) + '\n')
