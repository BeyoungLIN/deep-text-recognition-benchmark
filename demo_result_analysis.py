import re
from collections import defaultdict

# shufa_pic/square_img_dev/龠~美工-崩雲體~219730.jpg	龠                        	0.9982


def get_wrong_dict(file_handle):
    wrong_dict = dict()
    for line in file_handle:
        if line.startswith('shufa_pic/square_img_dev'):
            line = line.strip().split('\t')
            file_name = line[0]
            pred_answer = line[1].strip()
            file_name = file_name.split('/')[2]
            gt_answer = file_name[0]
            if pred_answer != gt_answer:
                wrong_dict[file_name] = pred_answer
    return wrong_dict


if __name__ == '__main__':
    # with open('result\\demo_s.log', 'r', encoding='utf-8') as s_log:
    #     s_wrong_list = get_wrong_list(s_log)
    # for i in s_wrong_list:
    #     print(i)

    with open('result\\demo_m.log', 'r', encoding='utf-8') as m_log:
        m_wrong_dict = get_wrong_dict(m_log)
    with open('result\\demo_l.log', 'r', encoding='utf-8') as l_log:
        l_wrong_dict = get_wrong_dict(l_log)

    pattern = re.compile('(.)~(.+)~(\\d+).jpg')
    # both wrong
    bw_key = m_wrong_dict.keys() & l_wrong_dict.keys()
    print('M字符集和L字符集均错误个数：' + str(len(bw_key)))
    print(bw_key)

    mwlr_key = m_wrong_dict.keys() - l_wrong_dict.keys()
    print('M字符集错误，L字符集正确个数：' + str(len(mwlr_key)))
    print(mwlr_key)

    mrlw_key = l_wrong_dict.keys() - m_wrong_dict.keys()
    print('M字符集正确，L字符集错误个数：' + str(len(mrlw_key)))
    print(mrlw_key)

    print()
    print('错误统计：')
    wrong_writer_dict = defaultdict(int)
    for k in m_wrong_dict.keys():
        re_res = re.match(pattern, k)
        if re_res is not None:
            wrong_writer_dict[re_res[2]] += 1
    for k in l_wrong_dict.keys():
        re_res = re.match(pattern, k)
        if re_res is not None:
            wrong_writer_dict[re_res[2]] += 1
    wrong_writer_dict = sorted(wrong_writer_dict.items(), key=lambda d: d[1], reverse=True)
    print(wrong_writer_dict)

    xingshu_lishu_count = 0
    total_count = 0
    for writer, count in wrong_writer_dict:
        if '行書' in writer or '隸書' in writer:
            xingshu_lishu_count += count
        total_count += count
    print('行书隶书错误: %d' % xingshu_lishu_count)
    print('总错误：%d' % total_count)
    print('比例为: %.4f' % (xingshu_lishu_count / total_count))
