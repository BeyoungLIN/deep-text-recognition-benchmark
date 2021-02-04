# -*- encoding: utf-8 -*-
__author__ = 'Euphoria'

import os
import sys

import six
import argparse
import lmdb
from PIL import Image
import numpy as np
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_env1', type=str)
    parser.add_argument('input_env2', type=str)
    parser.add_argument('output_env', type=str)
    parser.add_argument('--map_size', type=int, default=1099511627776, help='lmdb dataset size')
    args = parser.parse_args()
    return args


def get_item(env, index):
    with env.begin(write=False) as txn:
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key).decode('utf-8')
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
    return imgbuf, label


def get_cnt(env):
    try:
        with env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode(), default=0))
            cnt = nSamples + 1
        print(f'out_cnt start from {cnt}')
    except lmdb.NotFoundError:
        cnt = 1
    return cnt


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)
    return


if __name__ == '__main__':
    args = parse_args()
    env1_path = args.input_env1
    env2_path = args.input_env2
    env_out_path = args.output_env

    input_env1 = lmdb.open(env1_path, map_size=args.map_size)
    input_cnt1 = get_cnt(input_env1)
    print('env1: {} samples'.format(input_cnt1))
    input_env2 = lmdb.open(env2_path, map_size=args.map_size)
    input_cnt2 = get_cnt(input_env2)
    print('env2: {} samples'.format(input_cnt2))

    os.makedirs(env_out_path, exist_ok=True)
    output_env = lmdb.open(env_out_path, map_size=args.map_size)
    output_cache = {}
    output_cnt = get_cnt(output_env)

    for idx in range(input_cnt1):
        idx += 1
        imgbuf, label = get_item(input_env1, idx)
        if not checkImageIsValid(imgbuf):
            continue
        imageKey = 'image-%09d'.encode() % output_cnt
        labelKey = 'label-%09d'.encode() % output_cnt
        output_cache[imageKey] = imgbuf
        output_cache[labelKey] = label.encode()
        if output_cnt % 1000 == 0:
            writeCache(output_env, output_cache)
            output_cache = {}
            print('Written %d files.' % (output_cnt))
        output_cnt += 1

    for idx in range(input_cnt2):
        idx += 1
        imgbuf, label = get_item(input_env2, idx)
        if not checkImageIsValid(imgbuf):
            continue
        imageKey = 'image-%09d'.encode() % output_cnt
        labelKey = 'label-%09d'.encode() % output_cnt
        output_cache[imageKey] = imgbuf
        output_cache[labelKey] = label.encode()
        if output_cnt % 1000 == 0:
            writeCache(output_env, output_cache)
            output_cache = {}
            print('Written %d files.' % (output_cnt))
        output_cnt += 1

    output_nSamples = output_cnt - 1
    output_cache['num-samples'.encode()] = str(output_nSamples).encode()
    writeCache(output_env, output_cache)
    print('Created dataset with %d samples' % output_nSamples)



