import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.contrib.layers.python.layers.layers import repeat
from fpn import ResFPN, _Conv2D


class _MakeHead(tfe.Network):
    def __init__(self, out_planes, kernel_size=3, strides=1):
        super(_MakeHead, self).__init__()
        self.conv1 = _Conv2D(256, kernel_size, strides)
        self.conv2 = _Conv2D(256, kernel_size, strides)
        self.conv3 = _Conv2D(256, kernel_size, strides)
        self.conv4 = _Conv2D(256, kernel_size, strides)

        self.conv_out = _Conv2D(out_planes, kernel_size, strides, activation=None)

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
        self.fpn = ResFPN()
        self.loc_head = _MakeHead(self.num_anchors * 4)
        self.cls_head = _MakeHead(self.num_anchors * self.num_classes)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        feature_maps = self.fpn(x)
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
