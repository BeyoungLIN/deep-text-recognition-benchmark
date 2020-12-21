""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """
import random

import fire
import argparse
import os
import lmdb
import cv2

import numpy as np


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


def createImageAndGt_Range_Dataset(inputPath, gtPath, train_outputPath, val_outputPath=None,
                                   checkValid=True, map_size=1099511627776, select_range=(0, 500000), train_ratio=0.995):
    """
    Create LMDB dataset for Imagenet type single char images' dataset.
    ARGS:
        inputPath  : input folder path where starts imagePath
        gtPath     : gt folder
        outputPath : LMDB output path
        checkValid : if true, check the validity of every image
        select_range: range of select index
    """
    global IMG_EXTENSIONS
    os.makedirs(train_outputPath, exist_ok=True)
    train_env = lmdb.open(train_outputPath, map_size=map_size)
    train_cache = {}
    train_cnt = 1
    if val_outputPath is not None:
        os.makedirs(val_outputPath, exist_ok=True)
        val_env = lmdb.open(val_outputPath, map_size=map_size)
        val_cache = {}
        val_cnt = 1

    filenames = []
    labels = []

    for i in range(*select_range):
        img_path = os.path.join(inputPath, str(i) + '.jpg')
        gt_path = os.path.join(gtPath, str(i) + '.txt')
        if os.path.isfile(img_path) and os.path.isfile(gt_path):
            filenames.append(img_path)
            labels.append(open(gt_path, 'r', encoding='utf-8').read().strip())
        else:
            break

    assert len(filenames) == len(labels)

    nSamples = len(filenames)
    for i in range(nSamples):
        imagePath = filenames[i]
        label = labels[i]

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(train_outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue
        if val_outputPath is not None and random.random() < train_ratio:
            imageKey = 'image-%09d'.encode() % train_cnt
            labelKey = 'label-%09d'.encode() % train_cnt
            train_cache[imageKey] = imageBin
            train_cache[labelKey] = label.encode()

            if train_cnt % 1000 == 0:
                writeCache(train_env, train_cache)
                train_cache = {}
                print('Written %d train files.' % (train_cnt))
            train_cnt += 1
        else:
            imageKey = 'image-%09d'.encode() % val_cnt
            labelKey = 'label-%09d'.encode() % val_cnt
            val_cache[imageKey] = imageBin
            val_cache[labelKey] = label.encode()

            if val_cnt % 1000 == 0:
                writeCache(val_env, val_cache)
                val_cache = {}
                print('Written %d val files.' % (val_cnt))
            val_cnt += 1

    train_nSamples = train_cnt-1
    train_cache['num-samples'.encode()] = str(train_nSamples).encode()
    writeCache(train_env, train_cache)
    if val_outputPath is not None:
        val_nSamples = val_cnt - 1
        val_cache['num-samples'.encode()] = str(val_nSamples).encode()
        writeCache(val_env, val_cache)

    if val_outputPath is not None:
        print('Created dataset with %d train samples, %d val samples' % (train_nSamples, val_nSamples))
    else:
        print('Created dataset with %d train samples' % train_nSamples)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='input folder path where starts imagePath')
    parser.add_argument('--gt_path', type=str, help='list of image path and label')
    parser.add_argument('--train_output_path', type=str, required=True, help='output folder path where store lmdb dataset')
    parser.add_argument('--val_output_path', type=str, default=None, help='output folder path where store lmdb dataset, if None, only create train')
    parser.add_argument('--check_valid', action='store_true', help='if true, check the validity of every image')
    parser.add_argument('--map_size', type=int, default=1099511627776, help='lmdb dataset size')
    parser.add_argument('--select_range', type=str, default='0-500000', help='select range')
    parser.add_argument('--train_ratio', type=float, default=0.995)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    select_range = args.select_range
    select_range = tuple(map(int, select_range.split('-')))
    assert len(select_range) == 2
    createImageAndGt_Range_Dataset(args.input_path, args.gt_path, args.train_output_path, args.val_output_path,
                                   args.check_valid, args.map_size, select_range, args.train_ratio)
