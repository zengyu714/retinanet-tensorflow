"""Encode object boxes and labels."""
import math
import tensorflow as tf
from utils.box import meshgrid, box_iou, box_nms, change_box_order
from configuration import conf


def _make_list_input_size(input_size):
    input_size = [input_size] * 2 if isinstance(input_size, int) else input_size
    return tf.cast(input_size, tf.float32)


class BoxEncoder:
    def __init__(self):
        # TODO
        # NOTE anchor areas should change according to the ACTUAL object's size
        # Otherwise the height and width of anchor would be out of tune
        # E.g., when the input is 448 x 448, object size ranges in []
        # anchor_areas might be [14^2, 28^2, 56^2, 112^2, 224^2]
        self.anchor_areas = [14 * 14., 28 * 28., 56 * 56., 112 * 112., 224 * 224.]  # p3 -> p7
        self.aspect_ratios = [1 / 2., 1 / 1., 2 / 1.]
        self.scale_ratios = [1., pow(2, 1 / 3.), pow(2, 2 / 3.)]
        self.anchor_wh = self._get_anchor_wh()

    def _get_anchor_wh(self):
        """Compute anchor width and height for each feature map.

        Returns:
            anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        """
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h = ar
                h = math.sqrt(s / ar)
                w = ar * h
                for sr in self.scale_ratios:  # scale
                    anchor_h = h * sr
                    anchor_w = w * sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return tf.reshape(anchor_wh, [num_fms, -1, 2])  # shape [5, 9(3x3), 2]

    def _get_anchor_boxes(self, input_size):
        """Compute anchor boxes for each feature map.
        Args:
            input_size: (list) model input size of (w, h)

        Returns:
            boxes: (list) anchor boxes for each feature map. Each of size [#anchors, 4],
                          where #anchors = fmw * fmh * #anchors_per_cell
        """
        num_fms = len(self.anchor_areas)
        fm_sizes = [(tf.ceil(input_size[0] / pow(2., i + 3)), tf.ceil(input_size[1] / pow(2., i + 3)))
                    for i in range(num_fms)]  # TODO modify by p3 -> p7 feature map sizes
        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = tf.div(input_size, fm_size)
            fm_w, fm_h = [tf.cast(i, tf.int32) for i in [fm_size[0], fm_size[1]]]
            xy = tf.cast(meshgrid(fm_w, fm_h), tf.float32) + 0.5  # [fm_h*fm_w, 2]
            xy = tf.tile(tf.reshape((xy * grid_size), [fm_h, fm_w, 1, 2]), [1, 1, 9, 1])
            wh = tf.tile(tf.reshape(self.anchor_wh[i], [1, 1, 9, 2]), [fm_h, fm_w, 1, 1])
            box = tf.concat([xy, wh], 3)  # [x, y, w, h]
            boxes.append(tf.reshape(box, [-1, 4]))
        return tf.concat(boxes, 0)

    def encode(self, boxes, labels, input_size):
        """Encode target bounding boxes and class labels.

        We obey the Faster RCNN box coder:
            tx = (x - anchor_x) / anchor_w
            ty = (y - anchor_y) / anchor_h
            tw = log(w / anchor_w)
            th = log(h / anchor_h)

        Args:
            boxes: (tensor) bounding boxes of (xmin, ymin, xmax, ymax), sized [#obj, 4].
            labels: (tensor) object class labels, sized [#obj, ].
            input_size: (int/tuple) model input size of (w, h), should be the same.
        Returns:
            loc_trues: (tensor) encoded bounding boxes, sized [#anchors, 4].
            cls_trues: (tensor) encoded class labels, sized [#anchors, ].
        """
        input_size = _make_list_input_size(input_size)
        boxes = tf.reshape(boxes, [-1, 4])
        anchor_boxes = self._get_anchor_boxes(input_size)

        boxes = change_box_order(boxes, 'xyxy2xywh')
        boxes *= tf.tile(input_size, [2])  # scaled back to original size

        ious = box_iou(anchor_boxes, boxes, order='xywh')
        max_ids = tf.argmax(ious, axis=1)
        max_ious = tf.reduce_max(ious, axis=1)

        boxes = tf.gather(boxes, max_ids)  # broadcast automatically, [#anchors, 4]

        loc_xy = (boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]
        loc_wh = tf.log(boxes[:, 2:] / anchor_boxes[:, 2:])
        loc_trues = tf.concat([loc_xy, loc_wh], 1)
        cls_trues = tf.gather(labels, max_ids)  # TODO: check if needs add 1 here
        cls_trues = tf.where(max_ious < 0.5, tf.zeros_like(cls_trues), cls_trues)
        ignore = (max_ious > 0.4) & (max_ious < 0.5)  # ignore ious between (0.4, 0.5), and marked as -1
        cls_trues = tf.where(ignore, tf.ones_like(cls_trues) * -1, cls_trues)
        cls_trues = tf.cast(cls_trues, tf.float32)
        return loc_trues, cls_trues

    def decode(self, loc_preds, cls_preds, input_size,
               cls_thred=conf.cls_thred, nms_thred=conf.nms_thred, return_score=False):
        """Decode outputs back to bouding box locations and class labels.

        Args:
            loc_preds: (tensor) predicted locations, sized [#anchors, 4].
            cls_preds: (tensor) predicted class labels, sized [#anchors, #classes].
            input_size: (int/tuple) model input size of (w, h), should be the same.
            cls_thred: class score threshold
            nms_thred: non-maximum suppression threshold
            return_score: (bool) indicate whether to return score value.
        Returns:
            boxes: (tensor) decode box locations, sized [#obj, 4]. [xmin, ymin, xmax, ymax]
            labels: (tensor) class labels for each box, sized [#obj, ].
        """
        assert len(loc_preds.get_shape().as_list()) == 2, 'Ensure the location input shape to be [#anchors, 4]'
        assert len(cls_preds.get_shape().as_list()) == 2, 'Ensure the class input shape to be [#anchors, #classes]'

        input_size = _make_list_input_size(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)

        loc_xy = loc_preds[:, :2]
        loc_wh = loc_preds[:, 2:]

        xy = loc_xy * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
        wh = tf.exp(loc_wh) * anchor_boxes[:, 2:]
        boxes = tf.concat([xy - wh / 2, xy + wh / 2], 1)  # [#anchors, 4]

        labels = tf.argmax(cls_preds, 1)  # [#anchors, ]
        score = tf.reduce_max(tf.sigmoid(cls_preds), 1)

        ids = tf.cast(score > cls_thred, tf.int32)
        ids = tf.squeeze(tf.where(tf.not_equal(ids, 0)))
        # ids.nonzero().squeeze()  # [#obj, ]
        if not ids.get_shape().as_list():  # Fail to detect, choose the max score
            ids = tf.argmax(score)
        keep = box_nms(tf.gather(boxes, ids), tf.gather(score, ids), threshold=nms_thred)

        def _index(t, index):
            """Gather tensor successively
            E.g., _index(boxes, [idx_1, idx_2]) = tf.gather(tf.gather(boxes, idx_1), idx_2)
            """
            if not isinstance(index, (tuple, list)):
                index = list(index)
            for i in index:
                t = tf.gather(t, i)
            return t

        if return_score:
            return _index(boxes, [ids, keep]), _index(labels, [ids, keep]), _index(score, [ids, keep])
        return _index(boxes, [ids, keep]), _index(labels, [ids, keep])


def test():
    input_size = (448, 448)
    dataset = dataset_generator('val', input_size, 1, 16, 100)
    box_encoder = BoxEncoder()
    model = RetinaNet()

    with tf.device("gpu:0"):
        for i, (image, loc_trues, cls_trues) in enumerate(tfe.Iterator(dataset)):
            print('loc_trues shape: {}'.format(loc_trues.shape))
            print('cls_trues shape: {}'.format(cls_trues.shape))
            loc_preds, cls_preds = model(image)

            with tf.device("cpu:0"):
                # Decode one by one in a batch
                loc_preds, cls_preds = box_encoder.decode(loc_preds.cpu()[0], cls_preds.cpu()[0], input_size)
                print('loc_preds {} shape: {}'.format(loc_preds, loc_preds.shape))
                print('cls_preds {} shape: {}'.format(cls_preds, cls_preds.shape))
            break


if __name__ == "__main__":
    import tensorflow.contrib.eager as tfe
    from inputs import dataset_generator
    from retinanet import RetinaNet

    tfe.enable_eager_execution()
    test()
