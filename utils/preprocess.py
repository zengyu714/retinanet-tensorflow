import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from utils import image_ops


def distort_intensity(image, scope=None):
    with tf.name_scope(scope, 'distort_intensity', [image]):
        prob = tf.random_uniform([], 0, 1.0)
        do_cond = tf.less(prob, .5)
        image = control_flow_ops.cond(do_cond,
                                      lambda: tf.image.random_brightness(image, max_delta=32. / 255.),
                                      lambda: image)
        image = control_flow_ops.cond(do_cond,
                                      lambda: tf.image.random_contrast(image, lower=0.7, upper=1.3),
                                      lambda: image)
        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(image, bboxes, labels, out_shape, scope='preprocessing_train'):
    with tf.name_scope(scope, 'preprocessing_train', [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        # Convert to float scaled [0, 1].
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # Resize image to output size.
        image = tf.image.resize_images(image, out_shape)

        # Randomly flip the image horizontally.
        image, bboxes = image_ops.random_flip_lr(image, bboxes)

        # Randomly distort the black-and-white intensity.
        image = distort_intensity(image)

        # H x W x C --> C x H x W
        image = tf.transpose(image, perm=(2, 0, 1))
        return image, bboxes, labels


def preprocess_for_val(image, bboxes, labels, out_shape, scope='preprocessing_eval'):
    with tf.name_scope(scope):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        # Convert to float scaled [0, 1].
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # Resize image to output size.
        image = tf.image.resize_images(image, out_shape)

        # H x W x C --> C x H x W
        image = tf.transpose(image, perm=(2, 0, 1))
        return image, bboxes, labels


def preprocess(image, bboxes, labels, mode, out_shape=(300, 300)):
    """Pre-process an given image.
       NOTE: Default use NxCxHxW, 'channels_first' is typically faster on GPUs
    Args:
      image: A `Tensor` representing an image of arbitrary size.
      bboxes: (list) bounding boxes [xmin, ymin, xmax, ymax]
      labels: (int) corresponding label in [1, 2, ..., #num_classes]
      out_shape: The height and width of the image after preprocessing.
      mode: `train` if we're preprocessing the image for training and `val` for evaluation.

    Returns:
      A preprocessed image.
    """
    func = eval('preprocess_for_' + mode)
    return func(image, bboxes, labels, out_shape=out_shape)
