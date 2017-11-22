import tensorflow as tf
import tensorflow.contrib.eager as tfe

from fpn import ResFPN


class _MakeHead(tfe.Network):
    def __init__(self, out_planes, kernel_size=3, strides=1, head_name=''):
        super(_MakeHead, self).__init__(name='')

        def conv2d(filters, kernel_size, strides=1, activation=tf.nn.relu, name=''):
            """Mainly use to freeze data_format and padding"""
            l = tf.layers.Conv2D(filters, kernel_size,
                                 strides=strides,
                                 activation=activation,
                                 padding='same',
                                 data_format='channels_first',
                                 name=name)
            return self.track_layer(l)

        self.conv1 = conv2d(256, kernel_size, strides, name=head_name + '_1')
        self.conv2 = conv2d(256, kernel_size, strides, name=head_name + '_2')
        self.conv3 = conv2d(256, kernel_size, strides, name=head_name + '_3')
        self.conv4 = conv2d(256, kernel_size, strides, name=head_name + '_4')

        self.conv_out = conv2d(out_planes, kernel_size, strides, activation=None, name=head_name + '_out')

    def call(self, x):
        x = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        return self.conv_out(x)


class RetinaNet(tfe.Network):
    """ RetinaNet defined in Focal loss paper
     See: https://arxiv.org/pdf/1708.02002.pdf
    """
    num_anchors = 9
    num_classes = 12

    def __init__(self):
        super(RetinaNet, self).__init__()
        """
        Args:
            num_classes: # of classification classes
            num_anchors: # of anchors in each feature map
        """

        def head_block(out_planes, name=None):
            l = _MakeHead(out_planes, head_name=name)
            return self.track_layer(l)

        def fpn_block():
            l = ResFPN()
            return self.track_layer(l)

        self.fpn = fpn_block()
        self.loc_head = head_block(self.num_anchors * 4, name='Location')
        self.cls_head = head_block(self.num_anchors * self.num_classes, name='Class')

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        feature_maps = self.fpn(x, training)
        loc_preds = []
        cls_preds = []
        for feature_map in feature_maps:
            loc_pred = self.loc_head(feature_map)
            cls_pred = self.cls_head(feature_map)
            # [N, 9*4, H, W] -> [N, H, W, 9*4] -> [N, H*W*9, 4]
            loc_pred = tf.reshape(tf.transpose(loc_pred, [0, 2, 3, 1]), [batch_size, -1, 4])
            cls_pred = tf.reshape(tf.transpose(cls_pred, [0, 2, 3, 1]), [batch_size, -1, self.num_classes])
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)

        return tf.concat(loc_preds, axis=1), tf.concat(cls_preds, axis=1)


def test():
    with tf.device("gpu:0"):
        image = tf.random_uniform([3, 1, 448, 448])
        model = RetinaNet()
        loc_preds, cls_preds = model(image)
        print('loc_preds shape: {}'.format(loc_preds.shape))
        print('cls_preds shape: {}'.format(cls_preds.shape))


if __name__ == "__main__":
    tfe.enable_eager_execution()

    test()
