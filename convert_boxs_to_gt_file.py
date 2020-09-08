
import json

with open('noise_data/text_lines_tags_vertical.txt', 'r', encoding='utf-8') as fp,\
        open('noise_data/gt.txt', 'w', encoding='utf-8') as op:
    for line in fp.readlines():
        split_res = line.strip().split('|')
        img_name = split_res[0]
        detail_json = split_res[1]
        detail_json = json.loads(detail_json)
        char_and_box_list = detail_json['char_and_box_list']
        chars = [it[0] for it in char_and_box_list]
        chars = ''.join(chars)
        op.write(img_name + '\t' + chars + '\n')