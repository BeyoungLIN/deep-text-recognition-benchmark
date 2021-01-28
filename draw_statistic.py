# -*- encoding: utf-8 -*-
__author__ = 'Euphoria'
import json
import matplotlib.pyplot as plt

from collections import defaultdict


if __name__ == '__main__':
    with open('result/extract.json', 'r', encoding='utf-8') as fp:
        total_length_map = json.load(fp)

    total_length_map = {int(k): v for k, v in total_length_map.items()}

    max_length = max(total_length_map.keys())

    print('max length: {}'.format(max_length))

    x = [_ for _ in range(1, max_length + 1)]
    y = [total_length_map.get(_, 0) for _ in range(1, max_length + 1)]

    avg = sum([k*v for k, v in total_length_map.items()]) / sum(total_length_map.values())
    print('avg length: {}'.format(avg))

    plt.figure()
    plt.bar(x, height=y)
    plt.grid(True)
    plt.xlabel('length')
    plt.ylabel('count')
    plt.savefig('./statictics.png')
    plt.show()