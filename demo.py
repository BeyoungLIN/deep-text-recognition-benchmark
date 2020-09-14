import string
import argparse

import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(image, text_for_pred, is_train=False)

                if opt.batch_max_length == 1:
                    # select top_k probabilty (greedy decoding) then decode index to character
                    k = opt.topk
                    preds = F.softmax(preds, dim=2)
                    topk = preds.topk(k)
                    topk_id = topk[1]
                    topk_prob = topk[0]
                    topk_id = topk_id.detach().cpu()[:, 0, :].unsqueeze(dim=1).numpy()  # (batch_size, topk)
                    # concat 3(['s']) to the end of ids
                    topk_s = np.ones_like(topk_id) * 3
                    topk_id = np.concatenate((topk_id, topk_s), axis=1)
                    topk_chars = converter.decode(topk_id, length_for_pred)
                    topk_probs = topk_prob.detach().cpu()[:, 0, :]  # (batch_size, topk)
                else:
                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, length_for_pred)

            if opt.batch_max_length == 1:
                log = open(f'./log_demo_result.csv', 'a', encoding='utf-8')
                # topk_probs = F.softmax(topk_probs, dim=-1)
                for img_name, pred, pred_max_prob in zip(image_path_list, topk_chars, topk_probs):
                    if 'Attn' in opt.Prediction:
                        pred = [p[:p.find('[s]')] for p in pred] # prune after "end of sentence" token ([s])
                    print(img_name, end='')
                    log.write(img_name)
                    for pred_char, pred_prob in zip(pred, pred_max_prob):
                        print(','+pred_char, end='')
                        print(',%.4f' % pred_prob, end='')
                        log.write(','+pred_char)
                        log.write(',%.4f' % pred_prob)
                    print()
                    log.write('\n')
                log.close()
            else:
                log = open(f'./log_demo_result.txt', 'a', encoding='utf-8')
                dashed_line = '-' * 80
                head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

                print(f'{dashed_line}\n{head}\n{dashed_line}')
                log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                    if 'Attn' in opt.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # calculate confidence score (= multiply of pred_max_prob)
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                    print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                    log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')
                log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    """ Output Setting """
    parser.add_argument('--topk', type=int, default=5, help='Top-k to output')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    '''
    if opt.character in ['CN-s', 'CN-m', 'CN-l']:
        if opt.character == 'CN-s':
            with open('charset/charset_s.txt', 'r', encoding='utf-8') as chars:
                charset = chars.readlines()
            charset = [c.strip() for c in charset]
        else:
            charset_csv = pd.read_csv('charset/all_abooks.unigrams_desc.Clean.rate.csv')
            if opt.character == 'CN-m':
                charset = charset_csv[['char']][charset_csv['acc_rate'] <= 0.999].values.squeeze(axis=-1).tolist()
            elif opt.character == 'CN-l':
                charset = charset_csv[['char']][charset_csv['acc_rate'] <= 0.9999].values.squeeze(axis=-1).tolist()
            else:
                raise ValueError
        charset = ''.join(charset)
        opt.character = charset
    elif opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    '''

    if opt.character.startswith('CN-'):
        if opt.character == 'CN-s':
            with open('charset/charset_s.txt', 'r', encoding='utf-8') as fp:
                charset = [c.strip() for c in fp.readlines()]
        elif opt.character == 'CN-m':
            with open('charset/charset_m.txt', 'r', encoding='utf-8') as fp:
                charset = [c.strip() for c in fp.readlines()]
        elif opt.character == 'CN-l':
            with open('charset/charset_l.txt', 'r', encoding='utf-8') as fp:
                charset = [c.strip() for c in fp.readlines()]
        else:
            raise ValueError
        charset = ''.join(charset)
        opt.character = charset
    elif opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    else:
        raise ValueError

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)
