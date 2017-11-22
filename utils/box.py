"""Some helper functions for PyTorch."""
import math
import tensorflow as tf

from numpy import argsort


def meshgrid(x, y, row_major=True):
    """Return meshgrid in range x & y.

    Args:
      x: (tf.int32) first dim range.
      y: (tf.int32) second dim range.
      row_major: (bool) row major or column major.

    Returns:
      (tf.Tensor) meshgrid, shape [x*y, 2]

    Example:
    >> meshgrid(3,2)
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    <tf.Tensor 'concat_1:0' shape=(6, 2) dtype=int32>

    >> meshgrid(3,2,row_major=False)
    0  0
    0  1
    0  2
    1  0
    1  1
    1  2
    <tf.Tensor 'concat_2:0' shape=(6, 2) dtype=int32>
    """
    xx = tf.reshape(tf.tile(tf.range(x), [y]), [-1, 1])
    yy = tf.reshape(tf.tile(tf.range(y), [x]), [-1, 1])
    return tf.concat([xx, yy], 1) if row_major else tf.concat([yy, xx], 1)


def change_box_order(boxes, order):
    """Change box order between (xmin, ymin, xmax, ymax) and (xcenter, ycenter, width, height).

    Args:
      boxes: (tf.tensor) bounding boxes, sized [N, 4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.

    Returns:
      (tf.tensor) converted bounding boxes, sized [N, 4].
    """
    assert order in ['xyxy2xywh', 'xywh2xyxy']
    a = boxes[:, :2]
    b = boxes[:, 2:]
    if order == 'xyxy2xywh':
        return tf.concat([(a + b) / 2, b - a], 1)
    return tf.concat([a - b / 2, a + b / 2], 1)


def box_iou(box1, box2, order='xyxy'):
    """Compute the intersection over union of two set of boxes.
    The default box order is (xmin, ymin, xmax, ymax).

    Args:
      box1: (tf.tensor) bounding boxes, sized [A, 4].
      box2: (tf.tensor) bounding boxes, sized [B, 4].
      order: (str) box order, either 'xyxy' or 'xywh'.

    Return:
      (tensor) iou, sized [A, B].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if order == 'xywh':
        box1, box2 = [change_box_order(i, 'xywh2xyxy') for i in [box1, box2]]

    # A: #box1, B: #box2
    lt = tf.maximum(box1[:, None, :2], box2[:, :2])  # [A, B, 2], coordinates left-top
    rb = tf.minimum(box1[:, None, 2:], box2[:, 2:])  # [A, B, 2], coordinates right-bottom

    wh = tf.clip_by_value(rb - lt,  # [A, B, 2], only clip the minimum
                          clip_value_min=0, clip_value_max=tf.float32.max)
    inter = wh[:, :, 0] * wh[:, :, 1]  # [A, B]
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [A,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [B,]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def box_nms(bboxes, scores, threshold=0.5, mode='union'):
    """Non maximum suppression.

    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.

    Returns:
      keep: (tensor) selected indices.

    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = argsort(scores.numpy())[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        xx1 = tf.clip_by_value(tf.gather(x1, order[1:]), clip_value_min=x1[i], clip_value_max=tf.float32.max)
        yy1 = tf.clip_by_value(tf.gather(y1, order[1:]), clip_value_min=y1[i], clip_value_max=tf.float32.max)
        xx2 = tf.clip_by_value(tf.gather(x2, order[1:]), clip_value_min=x2[i], clip_value_max=tf.float32.max)
        yy2 = tf.clip_by_value(tf.gather(y2, order[1:]), clip_value_min=y2[i], clip_value_max=tf.float32.max)

        w = tf.clip_by_value(xx2 - xx1, clip_value_min=0, clip_value_max=tf.float32.max)
        h = tf.clip_by_value(yy2 - yy1, clip_value_min=0, clip_value_max=tf.float32.max)
        inter = w * h

        if mode == 'union':
            ovr = inter / (areas[i] + tf.gather(areas, order[1:]) - inter)
        elif mode == 'min':
            ovr = inter / tf.clip_by_value(tf.gather(areas, order[1:]),
                                           clip_value_min=tf.float32.min, clip_value_max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = tf.squeeze(tf.where(tf.less_equal(ovr, threshold)))
        if len(ids.get_shape().as_list()) == 0:
            break
        order = order[ids + 1]
    return tf.cast(keep, tf.int32)
