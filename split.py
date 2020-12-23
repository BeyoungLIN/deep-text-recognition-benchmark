import os
import sys

import argparse
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    for img_path in os.listdir(args.input_dir):
        basename, ext = os.path.splitext(img_path)
        if ext not in ['.jpg', '.tif', '.jpeg', '.png']:
            continue
        output_path = os.path.join(args.output_dir, img_path)
        split_file = os.path.join(args.output_dir, basename + '.txt')
        img = cv2.imread(os.path.join(args.input_dir, img_path))
        for line in open(split_file, encoding='utf-8'):
            line = line.split(',')
            if len(line) % 2 == 1:
                line = line[:-1]
            line = list(map(int, line))
            cv2.line(img, (line[0], line[1]), (line[2], line[3]), color=(255, 0, 0), thickness=2)
        cv2.imwrite(output_path, img)