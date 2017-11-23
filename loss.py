import tensorflow as tf
import tensorflow.contrib.eager as tfe


def focal_loss(x, y, num_classes):
    """Focal loss.

    Args:
        x: (tensor) sized [N, D]
        y: (tensor) sized [N,]
        num_classes: numbers of classes
    Return:
      (tensor) focal loss.
    """
    alpha = 0.25
    gamma = 2

    y = tf.cast(y, tf.int32)
    t = tf.one_hot(y, depth=num_classes + 1)  # [N, #total_cls]
    t = t[:, 1:]  # exclude background

    p = tf.sigmoid(x)
    pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
    w = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
    w = w * tf.pow((1 - pt), gamma)
    return tf.losses.sigmoid_cross_entropy(t, x, w)


def focal_loss_alt(x, y, num_classes):
    """Focal loss alternative.

    Args:
        x: (tensor) sized [N, D]
        y: (tensor) sized [N,]
        num_classes: numbers of classes

    Return:
      (tensor) focal loss.
    """
    alpha = 0.25

    y = tf.cast(y, tf.int32)
    t = tf.one_hot(y, depth=num_classes + 1)  # [N, #total_cls]
    t = t[:, 1:]

    xt = x * (2 * t - 1)  # xt = x if t > 0 else -x
    pt = tf.log_sigmoid(2 * xt + 1)

    w = alpha * t + (1 - alpha) * (1 - t)
    loss = -w * pt / 2
    return tf.reduce_sum(loss)


def loss_fn(loc_preds, loc_trues, cls_preds, cls_trues, num_classes=12):
    """Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

    Args:
        loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
        loc_trues: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
        cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
        cls_trues: (tensor) encoded target labels, sized [batch_size, #anchors].

    loss:
        (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
    """

    # 1. loc_loss: tf.losses.huber_loss
    # ==================================================================
    # TODO: cannot use boolean_mask/slice between GPU and CPU
    mask_index = tf.where(cls_trues > 0)
    masked_loc_preds = tf.gather_nd(loc_preds, mask_index)  # [#valid_pos, 4]
    masked_loc_trues = tf.gather_nd(loc_trues, mask_index)  # [#valid_pos, 4]

    loc_loss = tf.losses.huber_loss(masked_loc_preds, masked_loc_trues)
    # ==================================================================

    # 2. cls_loss = FocalLoss(loc_preds, loc_trues)
    # ==================================================================
    mask_index = tf.where(cls_trues > -1)
    masked_cls_preds = tf.reshape(tf.gather_nd(cls_preds, mask_index), [-1, num_classes])  # [#valid_anchors, #cls]
    masked_cls_trues = tf.reshape(tf.gather_nd(cls_trues, mask_index), [-1])  # [#valid_anchors]

    # cls_loss = focal_loss_alt(masked_cls_preds, masked_cls_trues, num_classes)
    cls_loss = focal_loss(masked_cls_preds, masked_cls_trues, num_classes)
    # ==================================================================

    return loc_loss, cls_loss


def test():
    # with tf.device("/gpu:0"):
    # [batch_size, #anchors]s
    # loc_preds = tf.random_uniform([3, 10, 4])
    # loc_trues = tf.random_uniform([3, 10, 4])
    # cls_preds = tf.random_uniform([3, 10, 12])
    # cls_trues = tf.random_uniform([3, 10])

    from inputs import dataset_generator
    from retinanet import RetinaNet

    dataset = dataset_generator('train', (448, 448), 1, 16, 100)
    model = RetinaNet()

    with tf.device("/gpu:0"):
        for i, (image, loc_trues, cls_trues) in enumerate(tfe.Iterator(dataset)):
            loc_preds, cls_preds = model(image, training=True)
            loc_loss, cls_loss = loss_fn(loc_preds, loc_trues, cls_preds, cls_trues)
            print("Step 0: Location loss: {:.5f}  |  Class loss: {:.5f}".format(loc_loss.numpy(), cls_loss.numpy()))
            break


if __name__ == '__main__':
    tfe.enable_eager_execution()
    test()
