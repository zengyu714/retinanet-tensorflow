import numpy as np
import xml.etree.ElementTree as ET

from functools import partial

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.data import Dataset

from encoder import BoxEncoder
from configuration import conf
from utils.preprocess import preprocess


def get_name_list(txt_path):
    """Get the list of filenames indicates the index of image and annotation
        Args:
            txt_path: (str) path to the text file, typically is 'path/to/ImageSets/Main/trainval.txt'
        Returns:
            (list) of filenames: E.g., [000012, 000068, 000070, 000073, 000074, 000078, ...]
        """
    with tf.gfile.GFile(txt_path) as f:
        lines = f.readlines()
    return [l.strip() for l in lines]


def parse_anno_xml(xml_path):
    """Parse the annotation file (.xml)

    Args:
        xml_path: path to xml file
    Returns:
        bboxes: (list) contains normalized coordinates of [xmin, ymin, xmax, ymax]
        labels: (list) contains **int** index of corresponding class
    """
    root = ET.parse(xml_path).getroot()

    # image shape, [h, w, c]
    shape = [int(root.find('size').find(i).text) for i in ['height', 'width', 'depth']]
    height, width, _ = shape

    # annotations
    label_texts, labels, bboxes = [], [], []
    for obj in root.findall('object'):
        label_text = obj.find('name').text.lower().strip()
        label_texts.append(label_text.encode('utf8'))
        labels.append(conf.name_to_label_map[label_text])

        box = [int(obj.find('bndbox').find(p).text) for p in ['xmin', 'ymin', 'xmax', 'ymax']]
        box /= np.array([width, height] * 2)  # rescale boundingbox to [0, 1]
        bboxes.append(box.tolist())
    return bboxes, labels


def split_filename(mode):
    """Convert mode to split filename, that is,
    if mode == 'train', filename = 'trainval.txt'
    if mode == 'val',   filename = 'val.txt'
    """
    if mode == 'train':
        mode += 'val'
    return '{}/{}.txt'.format(conf.split_dir, mode)


def construct_naive_dataset(name_list):
    """Construct corresponding naive dataset

    Args:
        name_list: (list) names without format, say, ['000001', '000003']
    """
    impath_list = ['{}/{}.jpg'.format(conf.image_dir, p) for p in name_list]
    bboxes_list = [parse_anno_xml('{}/{}.xml'.format(conf.annot_dir, p))[0] for p in name_list]
    labels_list = [parse_anno_xml('{}/{}.xml'.format(conf.annot_dir, p))[1] for p in name_list]
    return Dataset.from_tensor_slices((impath_list, bboxes_list, labels_list))


def dataset_generator(mode,
                      input_size=conf.input_size,
                      num_epochs=conf.num_epochs,
                      batch_size=conf.batch_size,
                      buffer_size=conf.buffer_size,
                      return_iterator=False):
    """Create dataset including [image_dataset, bboxes_dataset, labels_dataset]
        Args:
            mode: (str) 'train' or 'val'
            input_size: (int) input size (h, w)
            num_epochs: (int) nums of looping over the dataset
            batch_size: (int) batch size for input
            buffer_size: (int) representing the number of elements from this dataset
                               from which the new dataset will sample, say, it
                               maintains a fixed-size buffer and chooses the next
                               element uniformly at random from that buffer
            return_iterator: (bool) if false, return dataset instead
    """
    assert mode in ['train', 'val'], "Unknown mode {} besides 'train' and 'val'".format(mode)

    # Helper function to decode image data and processing it
    # ==============================================================================
    def _decode_image(impath, bboxes, labels):
        im_raw = tf.read_file(impath)
        # convert to a grayscale image and downscale x2
        image = tf.image.decode_jpeg(im_raw, channels=1, ratio=2)
        # image.set_shape([None, None, 1])
        return image, bboxes, labels

    _preprocess = partial(preprocess, mode=mode, out_shape=input_size)

    def _encode_boxes(image, bboxes, labels):
        loc_target, cls_target = BoxEncoder().encode(bboxes, labels, input_size)
        return image, loc_target, cls_target

    # ==============================================================================

    name_list = get_name_list(split_filename(mode))
    if mode == 'train':
        np.random.shuffle(name_list)

    dataset = construct_naive_dataset(name_list)
    dataset = dataset.map(_decode_image)
    dataset = dataset.map(_preprocess)
    dataset = dataset.map(_encode_boxes)

    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat(num_epochs)
    batched_dataset = dataset.batch(batch_size)
    if return_iterator:
        iterator = batched_dataset.make_one_shot_iterator()
        return iterator.get_next()
    else:
        # image, labels, bboxes
        return batched_dataset


def test(mode='train'):
    tfe.enable_eager_execution()

    dataset = dataset_generator(mode, (448, 448), 1, 16, 100)
    for i, (image, loc_trues, cls_trues) in enumerate(tfe.Iterator(dataset)):
        print(image.shape, '-' * 30, "{}th's label: {} [{}]".format(i, np.unique(cls_trues.numpy()), cls_trues.shape))


if __name__ == '__main__':
    test()
