import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from glob import glob
from tqdm import tqdm
from tensorflow.python.data import Dataset

from encoder import BoxEncoder
from retinanet import RetinaNet
from configuration import conf
from utils.misc import merge_with_label

parser = argparse.ArgumentParser(description='TensorFlow RetinaNet Training')
parser.add_argument('--cuda-device', default=0, type=int, help='gpu device index')

args = parser.parse_args()

conf.batch_size = 64


def deploy_dataset_generator(deploy_root_dir='data/Deploy',
                             batch_size=conf.batch_size,
                             input_size=conf.input_size):
    def _decode_image(im_path):
        im_raw = tf.read_file(im_path)
        # convert to a grayscale image and downscale x2
        image = tf.image.decode_jpeg(im_raw, channels=1, ratio=2)
        return image

    def _preprocess(im):
        # Convert to float scaled [0, 1].
        if im.dtype != tf.float32:
            im = tf.image.convert_image_dtype(im, dtype=tf.float32)

        # Resize image to output size.
        im = tf.image.resize_images(im, input_size)

        # H x W x C --> C x H x W
        return tf.transpose(im, perm=(2, 0, 1))

    def _sort(p):
        """'data/Deploy/KLAC/KLAC0003/KLAC0003_86.jpg'
        ==> 'data/Deploy/KLAC/KLAC0003/KLAC0003_0086.jpg'
        """
        prefix, old_name = p.split('_')
        new_name = old_name.zfill(8)
        return '_'.join([prefix, new_name])

    frames_name_list = sorted(glob('{}/*/*/*.jpg'.format(deploy_root_dir)), key=_sort)
    dir_dataset = Dataset.from_tensor_slices(frames_name_list)
    img_dataset = dir_dataset.map(_decode_image)
    img_dataset = img_dataset.map(_preprocess)
    dataset = Dataset.zip((img_dataset, dir_dataset))
    return dataset.batch(batch_size), len(frames_name_list)


def decode_batch(batch_loc_preds,
                 batch_cls_preds,
                 input_size=conf.input_size,
                 tf_box_order=True):
    """Choose the most confident one from multiple (if any) predictions per image.
    Make sure each image only has one output (loc + cls)

    Args:
        batch_loc_preds: (tensor) predicted locations, sized [batch, #anchors, 4].
        batch_cls_preds: (tensor) predicted class labels, sized [batch, #anchors, #classes].
        input_size: (int/tuple) model input size of (w, h), should be the same.
        tf_box_order: (bool) True: [ymin, xmin, ymax, xmax]
                            False: [xmin, ymin, xmax, ymax]
    Returns:
        batch_loc: (tensor)  decode batch box locations, sized [batch, 4]. [y_min, x_min, y_max, x_max]
        batch_cls: (tensor) class label for each box, sized [batch, ]
        batch_scores: (tensor) score for each box, sized [batch, ]

    """
    batch_loc, batch_cls, batch_scores = [], [], []
    for i, (loc_preds, cls_preds) in enumerate(zip(batch_loc_preds, batch_cls_preds)):
        loc, cls, scores = self.decode(loc_preds, cls_preds,
                                       input_size,
                                       max_output_size=10,
                                       return_score=True,
                                       tf_box_order=tf_box_order)
        if scores.shape[0] == 0:
            return [None] * 3
        max_score_id = tf.argmax(scores)
        batch_loc.append(tf.gather(loc, max_score_id).numpy() / input_size[0])
        for item in ['cls', 'scores']:
            eval('batch_' + item).append(tf.gather(eval(item), max_score_id).numpy())
    return [tf.convert_to_tensor(item, dtype=tf.float32) for item in [batch_loc, batch_cls, batch_scores]]


assert tfe.num_gpus() > 0, 'Make sure the GPU device exists'
device_name = '/gpu:{}'.format(args.cuda_device)
print('\n==> ==> ==> Using device {}'.format(device_name))

dataset, dataset_size = deploy_dataset_generator()
model = RetinaNet()
box_encoder = BoxEncoder()


def save_deployment():
    with tfe.restore_variables_on_create(tf.train.latest_checkpoint(conf.checkpoint_dir)):
        # epoch = tfe.Variable(1., name='epoch')
        # print('==> ==> ==> Restore from epoch {}...\n'.format(epoch.numpy()))
        gs = tf.train.get_or_create_global_step()
        print('==> ==> ==> Restore from global step {}...\n'.format(gs.numpy()))

        deploy_results = []
        # batch images
        for im_batch, p_batch in tqdm(tfe.Iterator(dataset),
                                      total=dataset_size // conf.batch_size,
                                      unit=' batch({})'.format(conf.batch_size)):
            with tf.device(device_name):
                loc_preds, cls_preds = model(im_batch.gpu())
            with tf.device("cpu:0"):
                scale = tf.convert_to_tensor([*conf.image_size] * 2, dtype=tf.float32) / conf.input_size[0]

                # single image
                for i, (loc_pred, cls_pred) in enumerate(zip(loc_preds.cpu(), cls_preds.cpu())):
                    boxes, labels_idx, scores = box_encoder.decode(loc_pred, cls_pred, conf.input_size,
                                                                   return_score=True)
                    pred = []  # multiple boxes per image
                    for box, label_idx, score in zip(boxes, labels_idx, scores):
                        pt = (box * scale).numpy().astype(int)  # [ymin, xmin, ymax, xmax]
                        coords = (pt[1], pt[0]), pt[3] - pt[1] + 1, pt[2] - pt[0] + 1
                        label_name = conf.class_name[label_idx.numpy()]
                        pred.append({'class': label_name, 'score': score.numpy(), 'position': coords})
                    deploy_results.append({'index': p_batch[i].numpy().decode('utf-8'), 'prediction': pred})

        np.save(conf.deployment_save_dir, deploy_results)
        np.save('{}/deploy_results.npy'.format(conf.deployment_save_dir), deploy_results)



def main(_):
    print('\n==> ==> ==> Saving the results...')
    save_deployment()
    print('\n==> ==> ==> Merging with true labels...')
    merge_with_label()


def test_inputs():
    dataset = deploy_dataset_generator(batch_size=16)
    for i, (image, paths) in enumerate(tfe.Iterator(dataset)):
        print(i, image.shape)


if __name__ == '__main__':
    tfe.enable_eager_execution()
    # test_inputs()
    tf.app.run()
