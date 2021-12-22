# -*- coding: utf-8 -*-
# @Time   : 2021/12/23 01:31
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : exam_dingxiu_1222_data.py

import os
import random

pth = '/disks/sdg/euphoria/datasets/DingXiu_1201/imgs/'
files = os.listdir(pth)

random_index = [random.randint(0, 3690000) for i in range(100)]
# print(random_index)

target_pth = '/disks/sdg/euphoria/datasets/DingXiu_check_1201_data'
if not os.path.exists(target_pth):
    os.mkdir(target_pth)

for index in random_index:
    # print(index)
    file_pth = os.path.join(pth, files[index])
    os.system('cp ' + file_pth + ' ' + target_pth)
