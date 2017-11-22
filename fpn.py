# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ResNet50 model definition compatible with TensorFlow's eager execution.
Reference [Deep Residual Learning for Image
Recognition](https://arxiv.org/abs/1512.03385)
Adapted from tf.keras.applications.ResNet50. A notable difference is that the
model here outputs logits while the Keras model outputs probability.

Retina Feature Pyramid Network based on ResNet50
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import utils.layers as ul


class _IdentityBlock(tfe.Network):
    """_IdentityBlock is the block that has no conv layer at shortcut.
    Args:
      kernel_size: the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      data_format: data_format for the input ('channels_first' or
        'channels_last').
    """

    def __init__(self, kernel_size, filters, stage, block, data_format):
        super(_IdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv2a = self.track_layer(tf.layers.Conv2D(
                filters1, (1, 1), name=conv_name_base + '2a', data_format=data_format))
        self.bn2a = self.track_layer(
                tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a'))

        self.conv2b = self.track_layer(tf.layers.Conv2D(
                filters2, kernel_size, padding='same', data_format=data_format, name=conv_name_base + '2b'))
        self.bn2b = self.track_layer(
                tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b'))

        self.conv2c = self.track_layer(tf.layers.Conv2D(
                filters3, (1, 1), name=conv_name_base + '2c', data_format=data_format))
        self.bn2c = self.track_layer(
                tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c'))

    def call(self, inputs, training=False):
        x = self.conv2a(inputs)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += inputs
        return tf.nn.relu(x)


class _ConvBlock(tfe.Network):
    """_ConvBlock is the block that has a conv layer at shortcut.
    Args:
        kernel_size: the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        data_format: data_format for the input ('channels_first' or
          'channels_last').
        strides: strides for the convolution. Note that from stage 3, the first
         conv layer at main path is with strides=(2,2), and the shortcut should
         have strides=(2,2) as well.
    """

    def __init__(self, kernel_size, filters, stage, block, data_format, strides=(2, 2)):
        super(_ConvBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv2a = self.track_layer(tf.layers.Conv2D(
                filters1, (1, 1), strides=strides, name=conv_name_base + '2a', data_format=data_format))
        self.bn2a = self.track_layer(
                tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a'))

        self.conv2b = self.track_layer(tf.layers.Conv2D(
                filters2, kernel_size, padding='same', name=conv_name_base + '2b', data_format=data_format))
        self.bn2b = self.track_layer(
                tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b'))

        self.conv2c = self.track_layer(tf.layers.Conv2D(
                filters3, (1, 1), name=conv_name_base + '2c', data_format=data_format))
        self.bn2c = self.track_layer(
                tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c'))

        self.conv_shortcut = self.track_layer(tf.layers.Conv2D(
                filters3, (1, 1), strides=strides, name=conv_name_base + '1', data_format=data_format))
        self.bn_shortcut = self.track_layer(
                tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1'))

    def call(self, inputs, training=False):
        x = self.conv2a(inputs)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        shortcut = self.conv_shortcut(inputs)
        shortcut = self.bn_shortcut(shortcut, training=training)

        x += shortcut
        return tf.nn.relu(x)


class ResFPN(tfe.Network):
    """Instantiates the ResNet50 architecture.
    Args:
      data_format: format for the image. Either 'channels_first' or
        'channels_last'.  'channels_first' is typically faster on GPUs while
        'channels_last' is typically faster on CPUs. See
        https://www.tensorflow.org/performance/performance_guide#data_formats
      name: Prefix applied to names of variables created in the model.
      trainable: Is the model trainable? If true, performs backward
          and optimization after call() method.
      include_top: whether to include the fully-connected layer at the top of the
        network.
      pooling: Optional pooling mode for feature extraction when `include_top`
        is `False`.
        - `None` means that the output of the model will be the 4D tensor
            output of the last convolutional layer.
        - `avg` means that global average pooling will be applied to the output of
            the last convolutional layer, and thus the output of the model will be
            a 2D tensor.
        - `max` means that global max pooling will be applied.
      classes: optional number of classes to classify images into, only to be
        specified if `include_top` is True.
    Raises:
        ValueError: in case of invalid argument for data_format.
    """

    def __init__(self):
        super(ResFPN, self).__init__(name='')

        data_format = 'channels_first'
        bn_axis = 1

        def conv2d(filters, kernel_size, strides=1, activation=tf.nn.relu, name=None):
            """Mainly use to freeze data_format and padding"""
            l = tf.layers.Conv2D(filters, kernel_size,
                                 strides=strides,
                                 activation=activation,
                                 padding='same',
                                 data_format='channels_first',
                                 name=name)
            return self.track_layer(l)

        def conv2d_transpose(filters, kernel_size, strides=1, activation=tf.nn.relu, name=None):
            l = tf.layers.Conv2DTranspose(filters, kernel_size,
                                          strides=strides,
                                          activation=activation,
                                          padding='same',
                                          data_format='channels_first',
                                          name=name)
            return self.track_layer(l)

        def conv_block(filters, stage, block, strides=(2, 2)):
            l = _ConvBlock(3, filters, stage=stage, block=block, data_format=data_format, strides=strides)
            return self.track_layer(l)

        def id_block(filters, stage, block):
            l = _IdentityBlock(3, filters, stage=stage, block=block, data_format=data_format)
            return self.track_layer(l)

        # Bottom-up
        # ==========================================================================================================
        self.conv1 = conv2d(64, 7, strides=2, activation=None, name='conv1')
        self.bn_conv1 = self.track_layer(tf.layers.BatchNormalization(axis=bn_axis, name='bn_conv1'))
        self.max_pool = self.track_layer(tf.layers.MaxPooling2D((3, 3), strides=(2, 2), data_format=data_format))

        self.l2a = conv_block([64, 64, 256], stage=2, block='a', strides=(1, 1))
        self.l2b = id_block([64, 64, 256], stage=2, block='b')
        self.l2c = id_block([64, 64, 256], stage=2, block='c')

        self.l3a = conv_block([128, 128, 512], stage=3, block='a')
        self.l3b = id_block([128, 128, 512], stage=3, block='b')
        self.l3c = id_block([128, 128, 512], stage=3, block='c')
        self.l3d = id_block([128, 128, 512], stage=3, block='d')

        self.l4a = conv_block([256, 256, 1024], stage=4, block='a')
        self.l4b = id_block([256, 256, 1024], stage=4, block='b')
        self.l4c = id_block([256, 256, 1024], stage=4, block='c')
        self.l4d = id_block([256, 256, 1024], stage=4, block='d')
        self.l4e = id_block([256, 256, 1024], stage=4, block='e')
        self.l4f = id_block([256, 256, 1024], stage=4, block='f')

        self.l5a = conv_block([512, 512, 2048], stage=5, block='a')
        self.l5b = id_block([512, 512, 2048], stage=5, block='b')
        self.l5c = id_block([512, 512, 2048], stage=5, block='c')

        self.conv6 = conv2d(256, 3, strides=2, name='conv6')
        self.conv7 = conv2d(256, 3, strides=2, name='conv7')

        # Lateral
        # ===================================================================================
        self.latlayer1 = conv2d(256, 1, name='lateral1')
        self.latlayer2 = conv2d(256, 1, name='lateral2')
        self.latlayer3 = conv2d(256, 1, name='lateral3')

        # Top-down
        # ===============================================D===============s====================
        self.upsample1 = conv2d_transpose(256, 4, strides=2, name='upsample1')
        self.toplayer1 = conv2d(256, 3, name='top1')
        self.upsample2 = conv2d_transpose(256, 4, strides=2, name='upsample2')
        self.toplayer2 = conv2d(256, 3, name='top2')

    def call(self, x, training=False):
        # Bottom-up
        c1 = self.conv1(x)
        c1 = self.bn_conv1(c1, training=training)
        c1 = tf.nn.relu(c1)
        c1 = self.max_pool(c1)

        c2 = self.l2a(c1, training=training)
        c2 = self.l2b(c2, training=training)
        c2 = self.l2c(c2, training=training)

        c3 = self.l3a(c2, training=training)
        c3 = self.l3b(c3, training=training)
        c3 = self.l3c(c3, training=training)
        c3 = self.l3d(c3, training=training)

        c4 = self.l4a(c3, training=training)
        c4 = self.l4b(c4, training=training)
        c4 = self.l4c(c4, training=training)
        c4 = self.l4d(c4, training=training)
        c4 = self.l4e(c4, training=training)
        c4 = self.l4f(c4, training=training)

        c5 = self.l5a(c4, training=training)
        c5 = self.l5b(c5, training=training)
        c5 = self.l5c(c5, training=training)

        p6 = self.conv6(c5)
        p7 = self.conv7(p6)

        # Top-down
        p5 = self.latlayer1(c5)
        p4 = ul.add_forced(self.upsample1(p5), self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = ul.add_forced(self.upsample2(p4), self.latlayer3(c3))
        p3 = self.toplayer2(p3)

        return p3, p4, p5, p6, p7


# TODO Add track_layer
class MobileFPN(tfe.Network):
    def __init__(self):
        super(MobileFPN, self).__init__()

    def call(self, inputs):
        # Bottom-up layers
        c1 = ul.conv2d(inputs, 64, 7, strides=2)
        c1 = ul.sep_conv2d(c1, 64, 3, strides=1)

        c2 = ul.sep_conv2d(c1, 128, 3, strides=2)
        c2 = ul.sep_conv2d(c2, 128, 3, strides=1)

        c3 = ul.sep_conv2d(c2, 256, 3, strides=2)
        c3 = ul.sep_conv2d(c3, 256, 3, strides=1)

        c4 = ul.sep_conv2d(c3, 512, 3, strides=2)
        c4 = ul.sep_conv2d(c4, 512, 3, strides=1)
        c4 = ul.sep_conv2d(c4, 512, 3, strides=1)
        c4 = ul.sep_conv2d(c4, 512, 3, strides=1)
        c4 = ul.sep_conv2d(c4, 512, 3, strides=1)

        p5 = ul.conv2d(c4, 256, 3, strides=2)
        p6 = ul.conv2d(p5, 256, 3, strides=2)

        # Top-down
        p4 = ul.conv2d(c4, 256, 1, activation=None)
        u4 = tf.layers.conv2d_transpose(p4, 256, [4, 4], strides=[2, 2], padding='same', data_format='channels_first')
        p3 = ul.add_forced(u4, ul.conv2d(c3, 256, 1, activation=None))
        p3 = ul.conv2d(p3, 256, 3, activation=None)
        u3 = tf.layers.conv2d_transpose(p3, 128, [4, 4], strides=[2, 2], padding='same', data_format='channels_first')
        p2 = ul.add_forced(u3, ul.conv2d(c2, 128, 1, activation=None))
        p2 = ul.conv2d(p2, 256, 3, activation=None)

        return p2, p3, p4, p5, p6


def test():
    # model = MobileFPN()
    model = ResFPN()
    with tf.device("gpu:0"):
        inputs = tf.random_uniform([3, 1, 448, 448])
        p3, p4, p5, p6, p7 = model(inputs)
        print('p2 shape: {}'.format(p3.shape))
        print('p6 shape: {}'.format(p7.shape))


if __name__ == "__main__":
    tfe.enable_eager_execution()
    test()
