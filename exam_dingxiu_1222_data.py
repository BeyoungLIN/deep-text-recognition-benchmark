# -*- coding: utf-8 -*-
# @Time   : 2021/12/23 01:31
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : exam_dingxiu_1222_data.py

import os
import random


def check_whole_gt():
    # 主要用来检查本次生成的大致效果
    imgs_pth = '/disks/sdg/euphoria/datasets/DingXiu_1201/imgs/'
    gts_pth = '/disks/sdg/euphoria/datasets/DingXiu_1201/gts/'
    files = os.listdir(imgs_pth)

    random_index = [random.randint(0, 3690000) for i in range(3000)]
    # print(random_index)

    target_pth = '/disks/sdg/euphoria/datasets/DingXiu_check_1201_data'
    if not os.path.exists(target_pth):
        os.mkdir(target_pth)

    for index in random_index:
        # print(index)
        img_pth = os.path.join(imgs_pth, files[index])
        gt_pth = os.path.join(gts_pth, files[index])[:-4] + '.txt'
        with open(gt_pth, 'r', encoding='utf-8') as f:
            con = f.read()
        con = con.replace('\n', '')
        con = con.replace(' ', '')
        if con.endswith('一') or con.endswith('以') or con.endswith('二'):
            os.system('cp ' + img_pth + ' ' + target_pth)
            os.system('cp ' + gt_pth + ' ' + target_pth)

# def check_yi():

check_whole_gt()


