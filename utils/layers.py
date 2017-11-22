import tensorflow as tf
import tensorflow.contrib.eager as tfe


def conv2d(inputs, filters, kernel_size, strides=1, activation=tf.nn.relu, scope=None):
    """Mainly use to freeze data_format and padding"""

    return tf.layers.conv2d(inputs, filters, kernel_size,
                            strides=strides,
                            activation=activation,
                            padding='same',
                            data_format='channels_first',
                            name=scope)


def sep_conv2d(inputs, filters, kernel_size, strides=1, activation=tf.nn.relu, scope=None):
    """Mainly use to freeze data_format and padding"""

    return tf.layers.separable_conv2d(inputs, filters, kernel_size,
                                      strides=strides,
                                      activation=activation,
                                      padding='same',
                                      data_format='channels_first',
                                      name=scope)


def add_forced(x, y):
    """Add two feature maps.

    Args:
        x: (Variable) upsampled feature map.
        y: (Variable) lateral feature map.

    Returns:
        (Variable) added feature map.
    """
    h, w = y.get_shape()[2:]
    return x[:, :, :h, :w] + y


