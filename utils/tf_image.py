import tensorflow as tf
from tensorflow.python.framework import tensor_shape


def fix_image_flip_shape(image, result):
    """Set the shape to 3 dimensional if we don't know anything else.
    Args:
      image: original image size
      result: flipped or transformed image
    Returns:
      An image whose shape is at least None,None,None.
    """
    image_shape = image.get_shape()
    if image_shape == tensor_shape.unknown_shape():
        result.set_shape([None, None, None])
    else:
        result.set_shape(image_shape)
    return result


def get_dims(image):
    """Returns the dimensions of an image tensor.
    Args:
      image: A 3-D Tensor of shape `[height, width, channels]`.
    Returns:
      A list of `[height, width, channels]` corresponding to the dimensions of the
        input image.  Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(3).as_list()
        dynamic_shape = tf.unstack(tf.shape(image), 3)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def resize_image(image, size):
    """Resize an image."""

    with tf.name_scope('resize_image'):
        height, width, channels = get_dims(image)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size)  # bilinear interpolation
        image = tf.reshape(image, tf.stack([size[0], size[1], channels]))
        return image


def random_flip_lr(image, bboxes, seed=None):
    """Random flip left-right of an image and its bounding boxes."""

    def flip_bboxes(bboxes):
        """Flip bounding boxes coordinates.
        Args:
            bboxes: [xmin, ymin, xmax, ymax]
        """
        bboxes = tf.stack([1 - bboxes[:, 2], bboxes[:, 1],
                           1 - bboxes[:, 0], bboxes[:, 3]], axis=-1)
        return bboxes

    # Random flip. Tensorflow implementation.
    with tf.name_scope('random_flip_left_right'):
        image = tf.convert_to_tensor(image, name='image')
        uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)
        mirror_cond = tf.less(uniform_random, .5)
        # Flip image.
        result = tf.cond(mirror_cond,
                         lambda: tf.reverse_v2(image, [1]),
                         lambda: image)
        # Flip bboxes.
        bboxes = tf.cond(mirror_cond,
                         lambda: flip_bboxes(bboxes),
                         lambda: bboxes)
        return fix_image_flip_shape(image, result), bboxes
