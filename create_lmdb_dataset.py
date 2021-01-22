

import os
import sys

import argparse
import lmdb
import cv2
import random
from tqdm import tqdm

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


def createDataset(inputPath, gtFile, outputPath, checkValid=True, map_size=1099511627776):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=map_size)
    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    for i in range(nSamples):
        imagePath, label = datalist[i].strip('\n').split('\t')
        imagePath = os.path.join(inputPath, imagePath)

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
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def createImagenetDataset(inputPath, outputPath, checkValid=True, map_size=1099511627776):
    """
    Create LMDB dataset for Imagenet type single char images' dataset.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        checkValid : if true, check the validity of every image
    """
    global IMG_EXTENSIONS
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=map_size)
    cache = {}
    cnt = 1

    filenames = []
    labels = []

    for root, subdirs, files in os.walk(inputPath, topdown=False):
        rel_path = os.path.relpath(root, inputPath) if (root != inputPath) else ''
        label = os.path.basename(rel_path)
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in IMG_EXTENSIONS:
                filenames.append(os.path.join(root, f))
                labels.append(label)

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
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def createDataset_ICDAR2019(inputPath, outputPath, train_and_eval=True, ratio=0.05, checkValid=True, map_size=1099511627776):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    if train_and_eval:
        os.makedirs(os.path.join(outputPath, 'train'), exist_ok=True)
        os.makedirs(os.path.join(outputPath, 'val'), exist_ok=True)

    if train_and_eval:
        train_env = lmdb.open(os.path.join(outputPath, 'train'), map_size=map_size)
        val_env = lmdb.open(os.path.join(outputPath, 'val'), map_size=map_size)
        train_cache = {}
        val_cache = {}
        train_cnt = 1
        val_cnt = 1
    else:
        env = lmdb.open(outputPath, map_size=map_size)
        cache = {}
        cnt = 1

    img_list = []
    label_list = []

    for root, dirs, files in os.walk(inputPath):
        for file in files:
            filename, ext = os.path.splitext(file)
            if ext in IMG_EXTENSIONS:
                img_list.append(os.path.join(root, file))
                label_list.append(os.path.join(root, filename + '.txt'))

    nSamples = len(img_list)
    for i in tqdm(range(nSamples)):
        imagePath = img_list[i]
        labelPath = label_list[i]

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        if not os.path.exists(labelPath):
            print('%s does not exits' % labelPath)

        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        with open(labelPath, 'r', encoding='utf-8') as fp:
            label = fp.readline().rstrip()

        if train_and_eval:
            if random.random() > ratio:
                imageKey = 'image-%09d'.encode() % train_cnt
                labelKey = 'label-%09d'.encode() % train_cnt
                train_cache[imageKey] = imageBin
                train_cache[labelKey] = label.encode()

                if train_cnt % 1000 == 0:
                    writeCache(train_env, train_cache)
                    train_cache = {}
                    print('Written %d in train set' % (train_cnt))
                train_cnt += 1
            else:
                imageKey = 'image-%09d'.encode() % val_cnt
                labelKey = 'label-%09d'.encode() % val_cnt
                val_cache[imageKey] = imageBin
                val_cache[labelKey] = label.encode()

                if val_cnt % 1000 == 0:
                    writeCache(val_env, val_cache)
                    val_cache = {}
                    print('Written %d in val set' % val_cnt)
                val_cnt += 1
        else:
            imageKey = 'image-%09d'.encode() % cnt
            labelKey = 'label-%09d'.encode() % cnt
            cache[imageKey] = imageBin
            cache[labelKey] = label.encode()

            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d / %d' % (cnt, nSamples))
            cnt += 1
    if train_and_eval:
        nSamples = train_cnt - 1
        train_cache['num-samples'.encode()] = str(nSamples).encode()
        writeCache(train_env, train_cache)
        print('Created train dataset with %d samples' % nSamples)
        nSamples = val_cnt - 1
        val_cache['num-samples'.encode()] = str(nSamples).encode()
        writeCache(val_env, val_cache)
        print('Created train dataset with %d samples' % nSamples)
    else:
        nSamples = cnt - 1
        cache['num-samples'.encode()] = str(nSamples).encode()
        writeCache(env, cache)
        print('Created dataset with %d samples' % nSamples)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', required=True, choices=['normal', 'imagenet', 'icdar2019'],
                        help='normal(imgs & gt_file) or imagenet(imgs in dirs)')
    parser.add_argument('--input_path', type=str, required=True, help='input folder path where starts imagePath')
    parser.add_argument('--gt_path', type=str, help='list of image path and label')
    parser.add_argument('--output_path', type=str, required=True, help='output folder path where store lmdb dataset')
    parser.add_argument('--check_valid', action='store_true', help='if true, check the validity of every image')
    parser.add_argument('--map_size', type=int, default=1099511627776, help='lmdb dataset size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    if args.type == 'normal':
        createDataset(args.input_path, args.gt_path, args.output_path, args.check_valid, args.map_size)
    elif args.type == 'imagenet':
        createImagenetDataset(args.input_path, args.output_path, args.check_valid, args.map_size)
    elif args.type == 'icdar2019':
        createDataset_ICDAR2019(args.input_path, args.output_path, checkValid=args.check_valid, map_size=args.map_size)
    else:
        raise ValueError('type should be normal or imagenet.')
