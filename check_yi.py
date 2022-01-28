# -*- coding: utf-8 -*-
# @Time   : 2021/9/11 11:40
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : check_yi.py

import os

def find_yi_list(check_path):
    res_list = []
    files = os.listdir(check_path)
    for file in files:
        gt_path = os.path.join(check_path, file)
        # print(gt_path)
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_txts = f.readlines()
        for gt_txt in gt_txts:
            gt_txt = gt_txt.replace('\n', '')
            gt_txt = gt_txt.replace(' ', '')
            # print(gt_txt)
            # if gt_txt.endswith('以') or gt_txt.endswith('一'):
            if gt_txt.endswith('一'):
                res_list.append(gt_path)
                print(gt_path)
    return res_list


def find_yi_list_one_file(file_path):
    res_list = []
    # files = os.listdir(file_path)
    # for file in files:
    #     gt_path = os.path.join(check_path, file)
        # print(gt_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        gt_txts = f.readlines()
    for gt_txt in gt_txts:
        gt_path = gt_txt.split('|')[0]
        gt_txt_ = gt_txt.replace('\n', '')
        gt_txt_ = gt_txt_.replace(' ', '')
        # print(gt_txt)
        # if gt_txt.endswith('以') or gt_txt.endswith('一'):
        if gt_txt_.endswith('一'):
            res_list.append(gt_txt)
            print(gt_path)
    return res_list


if __name__ == '__main__':
    # path = '/Users/Beyoung/Desktop/Projects/corpus/test/'
    # path = '/home/euphoria/deep-text-recognition-benchmark/dataset/DingXiu_train_output/gts'
    path = '/home/euphoria/deep-text-recognition-benchmark/dataset/Dingxiu_clean/val_combine.txt'
    # res = find_yi_list(path)
    res = find_yi_list_one_file(path)
    # print(res)
    with open ('gt_check_yi_val_一.txt', 'w', encoding='utf-8') as log:
        log.write(str(res))
    print(len(res))