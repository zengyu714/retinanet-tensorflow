import os
import codecs
import datetime
import numpy as np
from glob import glob
from tqdm import tqdm

from itertools import groupby
from configuration import conf
from sklearn.metrics import average_precision_score


def get_frame_prediction(d):
    """Get class name and index from the whole dictionary
    E.g.,
    '{'index': '...', 'prediction': [{'class': 'sp_sfb', 'position': ..., 'score': ...}]}], 'label': ...}'
    ==>  (dict) {'class': 'sp_sfb', 'position': ..., 'score': ...}
    """
    return d['prediction'][0]


def get_class(d):
    """Get class name from the whole dictionary
        E.g.,
        '{'index': 'data/Deploy/KLAC/KLAC0570/KLAC0570_12.jpg', 'prediction': ..., 'label': ...}'
        ==>  (str) 'KLAC'
        """
    return d['index'].split('/')[2]


def get_class_index(d):
    """Get class name and index from the whole dictionary
    E.g.,
    '{'index': 'data/Deploy/KLAC/KLAC0570/KLAC0570_12.jpg', 'prediction': ..., 'label': ...}'
    ==>  (str) 'KLAC0570'
    """
    return d['index'].split('/')[-2]


def get_video_index(d):
    """Get video index from the whole dictionary
    E.g.,
    '{'index': 'data/Deploy/KLAC/KLAC0570/KLAC0570_12.jpg', 'prediction': ..., 'label': ...}'
    ==>  (int) 570
    """
    return int(get_class_index(d)[-4:])


def get_frame_index(d):
    """Get frame index from the whole dictionary
    E.g.,
    '{'index': 'data/Deploy/KLAC/KLAC0570/KLAC0570_12.jpg', 'prediction': ..., 'label': ...}'
    ==>  (int) 12
    """
    return int(d['index'].split('_')[-1][:-4])


def label_map(line):
    if '非' in line:
        res = 'nsp_'
    elif '标准' in line:
        res = 'sp_'
    else:
        return 'others'

    if '上腹部' in line:
        res += 'sfb'
    elif '股骨' in line:
        res += 'gg'
    elif '丘脑' in line:
        res += 'qn'
    elif '侧脑室' in line:
        res += 'cns'
    elif '胆囊' in line:
        res += 'dn'
    elif '小脑' in line:
        res += 'xn'
    return res


def npy2txt(npy_dir):
    npy = np.load(npy_dir)
    txt_dir = npy_dir.replace('.npy', '.txt')
    with open(txt_dir, 'w+') as f:
        for n in tqdm(npy):
            f.write('{}\r\n'.format(n))


def convert_to_en_label(save_dir='data/Deploy/LABEL'):
    """Abbreviate Chinese annotation into the English mark, which is the same as label."""

    en_labels = []
    for cls in ['KLAC', 'KLFE', 'KLHC']:
        label_files = sorted(glob('{}/{}/*'.format(save_dir, cls)))
        for label_file in tqdm(label_files):
            with codecs.open(label_file, encoding='gbk') as f:
                lines = f.readlines()
                for line in lines:
                    # index: 'data/Deploy/KLAC/KLAC0001/KLAC0001_1.jpg', keep same with deployed index
                    index = '{}/{}'.format(label_file.replace('LABEL/', '').rstrip('.txt'), line.split(' ')[0])
                    label = label_map(line)
                    en_labels.append({'index': index, 'label': label})
    np.save(save_dir + '/{}.npy'.format('en_labels'), en_labels)


def separate_into_sp(interest_dir='deploy'):
    """Separate the whole merged dict into different files according to its class,
    say, interest_results ==> {KLAC, KLHC, KLFE}_interest_results"""

    interest = np.load('{}/interest_results.npy'.format(interest_dir))

    # split into separate videos #TODO separate to different class
    for cls, cls_frames in groupby(interest, key=get_class):
        np.save('{}/{}_interest_results.npy'.format(interest_dir, cls), list(cls_frames))


def correct_label(record):
    """Correct some mistakes in the manual labels"""
    from configuration import KLAC_CORRECTION, KLFE_CORRECTION, KLHC_CORRECTION

    video_idx = get_class_index(record)  # e.g., 'KLAC0570'
    frame_idx = get_frame_index(record)  # e.g., '12'
    cls = video_idx[:4]  # e.g., 'KLAC'

    true_right = [c for c in eval('{}_CORRECTION'.format(cls)) if c.startswith(video_idx)]
    prob_right = ['{}_{}.jpg'.format(video_idx, i) for i in range(frame_idx - 6, frame_idx + 7)]

    for tr in true_right:
        if tr in prob_right:
            return True
    return False


def strict_judgment(interest_path, threshold=0.8):
    """Do strict judgement from all interested predictions.
    That is, find one frame with the highest score in a video
    and then compare with label.

    Argument:
        interest_path: (str) path to file includes list of all interested records (dict)
    Return:
        y_truth: (list) true value in {0, 1} represents false and true sample respectively
        y_score: (list) probability ∈ [0, 1] with the same length as y_truth
    """
    preds, pred_hits = [], []

    interest = np.load(interest_path)
    cls = get_class(interest[0])[:4]

    # split into separate videos
    for vn, frames in groupby(interest, key=get_class_index):
        best = sorted(frames, key=lambda d: get_frame_prediction(d)['score'])[-1]
        # filter: the latter half videos are the real test set
        if get_video_index(best) > conf.class_nums[cls] // 2 and get_frame_prediction(best)['score'] > threshold:
            if best['label'] not in ['others', 'missed']:
                preds.append(best)

    cls_name = conf.label_name[cls]
    y_truth = [int(p['label'] == cls_name or correct_label(p)) for p in preds]
    y_score = [get_frame_prediction(p)['score'] for p in preds]

    # update corrected label
    corrected = []
    for i, p in enumerate(preds):
        if y_truth[i]:
            p['label'] = cls_name
        corrected.append(p)
    np.save(interest_path.replace('interest_results', 'corrected_best'), corrected)

    return y_truth, y_score


def merge_with_label():
    pred_path = 'deploy/deploy_results.npy'
    true_path = 'data/Deploy/LABEL/en_labels.npy'
    pred, true = [np.load(p) for p in [pred_path, true_path]]
    merged, i, j = [], 0, 0
    while i < len(pred) and j < len(true):
        p, t = pred[i], true[j]
        if p['index'] == t['index']:
            m = {**p, **t}
            merged.append(m)
            i += 1
            j += 1
        else:
            merged.append(p.update({'label': 'missed'}))
            i += 1

    # print('pred: {} | merged: {} | true: {}'.format(len(pred), len(merged), len(true)))
    assert len(pred) == len(merged), '#pred : {} is not equal with #merged{}'.format(len(pred), len(merged))
    np.save(pred_path.replace('deploy_', 'merged_'), merged)

    # filter the positive prediction
    interest = []
    for m in merged:
        try:
            tmp = get_frame_prediction(m)['class']
        except (TypeError, IndexError):
            tmp = 'none'
        if tmp in ['sp_qn', 'sp_gg', 'sp_sfb']:
            interest.append(m)
    np.save(pred_path.replace('deploy_', 'interest_'), interest)


def statistics(cls, threshold=0.5):
    """Compute statistic by frame"""
    with open('deploy/statistics.txt', 'a+') as f_stat:
        interest_path = 'deploy/{}_interest_results.npy'.format(cls)
        y_truth, y_score = strict_judgment(interest_path)

        y_truth = np.array(y_truth)
        y_pred = np.array(y_score) > threshold

        TP = sum(y_truth * y_pred)
        pred_nums, true_nums = sum(y_pred), sum(y_truth)
        nums = 'TP: {}\t# Pred: {}\t# True: {}\t'.format(TP, pred_nums, true_nums)

        # precision = precision_score(y_truth, y_pred)
        # recall = recall_score(y_truth, y_pred)
        # f1 = f1_score(y_truth, y_pred)
        # evaluation = 'Precision: {:.5f}\r\nRecall: {:.5f}\r\nF1 score: {:.5f}'.format(precision, recall, f1)

        ap = average_precision_score(y_truth, y_score)
        evaluation = 'AP: {:.4f}'.format(ap)
        now = datetime.datetime.now()
        info = """===> {} @ {}\n{}\r\n{}""".format(cls, now, nums, evaluation)
        f_stat.write(info)
        print(info)


if __name__ == '__main__':
    exp_root = '/home/zengyu/Lab/tensorflow/retinanet-tensorflow'
    os.chdir(exp_root)

    # convert_to_en_label()
    # separate_into_sp()
    for cls in ['KLAC', 'KLFE', 'KLHC']:
        statistics(cls)
