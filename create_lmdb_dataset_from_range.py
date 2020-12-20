""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

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


def createImageAndGt_Range_Dataset(inputPath, gtPath, outputPath,
                                   checkValid=True, map_size=1099511627776, select_range=(0, 500000)):
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
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=map_size)
    cache = {}
    cnt = 1

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', required=True, choices=['Normal', 'Imagenet', 'ImageAndGt'],
                        help='normal(imgs & gt_file) or imagenet(imgs in dirs)')
    parser.add_argument('--input_path', type=str, required=True, help='input folder path where starts imagePath')
    parser.add_argument('--gt_path', type=str, help='list of image path and label')
    parser.add_argument('--output_path', type=str, required=True, help='output folder path where store lmdb dataset')
    parser.add_argument('--check_valid', action='store_true', help='if true, check the validity of every image')
    parser.add_argument('--map_size', type=int, default=1099511627776, help='lmdb dataset size')
    parser.add_argument('--select_range', type=str, default='0-500000', help='select range')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    if args.type == 'Normal':
        createDataset(args.input_path, args.gt_path, args.output_path, args.check_valid, args.map_size)
    elif args.type == 'Imagenet':
        createImagenetDataset(args.input_path, args.output_path, args.check_valid, args.map_size)
    elif args.type == 'ImageAndGt':
        select_range = args.select_range
        select_range = list(map(int, select_range.split('-')))
        assert len(select_range) == 2
        createImageAndGt_Range_Dataset(args.input_path, args.gt_path, args.output_path, args.check_valid,
                                       args.map_size, select_range)
    else:
        raise ValueError('type should be normal or imagenet.')
