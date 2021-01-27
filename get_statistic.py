# -*- encoding: utf-8 -*-
__author__ = 'Euphoria'

import os
import sys

import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from pprint import pprint
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    length_map = defaultdict(int)
    for file in tqdm(os.listdir(args.path)):
        name, ext = os.path.splitext(file)
        if ext.lower() != '.txt':
            continue
        with open(os.path.join(args.path, file), 'r', encoding='utf-8') as fp:
            s = fp.readline().rstrip()
            length_map[len(s)] += 1
    pprint(length_map)
    keys = length_map.keys()
    max_length = max(*keys)
    x = [_ for _ in range(1, max_length+1)]
    y = [length_map[_] for _ in range(1, max_length+1)]

    plt.figure(figsize=(7, 5))
    plt.hist(y, bins=25)
    plt.grid(True)
    plt.xlabel('length')
    plt.ylabel('count')
    plt.savefig('./statictics.png')
