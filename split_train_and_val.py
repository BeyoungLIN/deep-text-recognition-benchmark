# -*- encoding: utf-8 -*-
__author__ = 'Euphoria'

import os
import sys

import argparse
import random
from tqdm import tqdm

import torch


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
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--train_output', type=str, default='result/train.txt')
    parser.add_argument('--val_output', type=str, default='result/val.txt')
    parser.add_argument('--val_ratio', type=float, default=0.05)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    done = get_already_done(args.done_path)
    cnt = get_count(args.count_path)
    todo_list = list()
    for root, dirs, files in os.walk(args.input_path):
        for file in files:
            if os.path.splitext(file)[-1].lower() not in IMG_EXT:
                continue
            file_path = os.path.join(root, file)
            if file_path in done:
                continue
            todo_list.append(file_path)
    print('total todo imgs count: {}'.format(len(todo_list)))
    random.shuffle(todo_list)
    offset = int(len(todo_list) * args.val_ratio)
    train_list = todo_list[offset:]
    val_list = todo_list[:offset]

    print('train imgs count: {}'.format(len(train_list)))
    with open(args.train_output, 'w', encoding='utf-8') as fp:
        for train_img in tqdm(train_list):
            fp.write(train_img + '\n')

    print('val imgs count: {}'.format(len(val_list)))
    with open(args.val_output, 'w', encoding='utf-8') as fp:
        for val_img in tqdm(val_list):
            fp.write(val_img + '\n')

