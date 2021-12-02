# -*- coding: utf-8 -*-
# @Time   : 2021/9/2 17:14
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : convert_fakepages_gt.py

import argparse
import os
import time


def convert_gt_folder(dir_path, output_path):
    files = os.listdir(dir_path)
    content = ''
    for file in files:
        file_path = os.path.join(dir_path, file)
        pic_name = file[:-4]
        char_label = ''
        with open(file_path, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                line = line.replace('\n', '')
                char_label += line.strip().split(',')[-1]
            content += pic_name + '\t' + char_label + '\n'


    with open(output_path, 'w') as op:
        # print(content)
        op.write(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True)
    args = parser.parse_args()
    # convert_gt_folder('/disks/sdb/euphoria/pkg/seg_detector/dataset/data/book_pages/test_gts', '/disks/sdb/euphoria/pkg/seg_detector/dataset/data/book_pages/test_gt.txt')
    # convert_gt_folder('/disks/sdb/euphoria/pkg/seg_detector/dataset/data/book_pages/train_gts', '/disks/sdb/euphoria/pkg/seg_detector/dataset/data/book_pages/train_gt.txt')
    # root_path = 'data/book_pages/'
    root_path = args.root_path
    print('start')
    val_time = time.time()
    convert_gt_folder(os.path.join(root_path, 'test_gts'), os.path.join(root_path, 'test_gt.txt'))
    print('val_set: Done')
    end_time = time.time()
    pro_time = end_time - val_time
    print(str(pro_time) + 's')
    convert_gt_folder(os.path.join(root_path, 'train_gts'), os.path.join(root_path, 'train_gt.txt'))
    print('train_set: Done')
    end_time_2 = time.time()
    pro_time_2 = end_time_2 - end_time
    print(str(pro_time_2) + 's')
    # convert_gt_folder('data/book_pages/test_gts', 'data/book_pages/test_gt.txt')
    # convert_gt_folder('data/book_pages/train_gts', 'data/book_pages/train_gt.txt')
#
# with open('noise_data/text_lines_tags_vertical.txt', 'r', encoding='utf-8') as fp,\
#         open('noise_data/gt.txt', 'w', encoding='utf-8') as op:
#     for line in fp.readlines():
#         split_res = line.strip().split('|')
#         img_name = split_res[0]
#         detail_json = split_res[1]
#         detail_json = json.loads(detail_json)
#         char_and_box_list = detail_json['char_and_box_list']
#         chars = [it[0] for it in char_and_box_list]
#         chars = ''.join(chars)
#         op.write(img_name + '\t' + chars + '\n')
