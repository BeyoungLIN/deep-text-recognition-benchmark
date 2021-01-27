# -*- encoding: utf-8 -*-
__author__ = 'Euphoria'

import os
import sys

import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from pprint import pprint
from tqdm import tqdm
import lmdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, type=str)
    args = parser.parse_args()
    return args


def get_lmdb_statistics(root):
    length_map = defaultdict(int)
    env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))
        for index in tqdm(range(nSamples)):
            index += 1  # lmdb starts with 1
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            length_map[len(label)] += 1
    return length_map


if __name__ == '__main__':
    args = parse_args()
    total_length_map = defaultdict(int)
    for lmdb_root in os.listdir(args.path):
        length_map = get_lmdb_statistics(os.path.join(args.path, lmdb_root))
        for k, v in length_map.items():
            total_length_map[k] += v
    pprint(total_length_map)
    max_length = max(total_length_map.keys())
    x = [_ for _ in range(1, max_length+1)]
    y = [total_length_map[_] for _ in range(1, max_length+1)]

    plt.figure()
    plt.hist(y)
    plt.grid(True)
    plt.xlabel('length')
    plt.ylabel('count')
    plt.savefig('./statictics.png')
