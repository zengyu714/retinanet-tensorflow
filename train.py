"""
Ref: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/mnist/mnist.py
"""
import os
import time
import argparse
import functools
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from loss import loss_fn
from inputs import dataset_generator
from retinanet import RetinaNet
from configuration import conf

parser = argparse.ArgumentParser(description='TensorFlow RetinaNet Training')
# parser.add_argument('--resume', '-r', default='store_true', help='resume from checkpoint')
parser.add_argument('--cuda-device', default=0, type=int, help='gpu device index')

args = parser.parse_args()


def train_one_epoch(model, optimizer, dataset, epoch):
    """Trains model on `dataset` using `optimizer`."""

    tf.train.get_or_create_global_step()

    def model_loss(images, loc_trues, cls_trues):
        loc_preds, cls_preds = model(images, training=True)
        loc_loss, cls_loss = loss_fn(loc_preds, loc_trues, cls_preds, cls_trues, num_classes=conf.num_class)
        total_loss = loc_loss + cls_loss
        tf.contrib.summary.scalar('loc_loss', loc_loss)
        tf.contrib.summary.scalar('cls_loss', cls_loss)
        tf.contrib.summary.scalar('total_loss', total_loss)
        return total_loss

    total_time = 0.
    for batch, (images, loc_trues, cls_trues) in enumerate(tfe.Iterator(dataset)):
        with tf.contrib.summary.record_summaries_every_n_global_steps(conf.log_interval):
            gs = tf.train.get_global_step()
            # Visualize the input images
            tf.contrib.summary.image('inputs', tf.transpose(images, [0, 2, 3, 1]), max_images=2, global_step=gs)
            # Optimize the loss
            batch_model_loss = functools.partial(model_loss, images, loc_trues, cls_trues)
            start = time.time()
            optimizer.minimize(batch_model_loss, global_step=gs)
            total_time += (time.time() - start)
            if batch % conf.log_interval == 0:
                time_in_ms = (total_time * 1000) / (batch + 1)
                print("[TRAINING] Batch: {}({:.0f}/{}) \t".format(batch, epoch, conf.num_epochs),
                      "total_loss: {:.6f} | avg_time: {:.2f}ms".format(batch_model_loss().numpy(), time_in_ms))


def validate(model, dataset, epoch):
    """Perform an evaluation of `model` on the examples from `dataset`."""

    batch_loss, avg_loss = 0., 0.
    start = time.time()
    for batch, (images, loc_trues, cls_trues) in enumerate(tfe.Iterator(dataset)):
        loc_preds, cls_preds = model(images)
        loc_loss, cls_loss = loss_fn(loc_preds, loc_trues, cls_preds, cls_trues, num_classes=conf.num_class)
        batch_loss += loc_loss + cls_loss
        avg_loss = batch_loss / (batch + 1)
        if batch % conf.log_interval == 0:
            fmt = [i.numpy() for i in [loc_loss, cls_loss, loc_loss + cls_loss, avg_loss]]
            print("[EVALUATION] Batch: {}({:.0f}/{})\t".format(batch, epoch, conf.num_epochs),
                  "loc_loss: {:.6f} | cls_loss: {:.6f} | total_loss: {:.6f} | avg_loss: {:.2f}".format(*fmt))

    time_in_ms = (time.time() - start) / 60
    print('[EVALUATION] Batch: {} Average time: {:.2f}min\n'.format(batch, time_in_ms))
    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('avg_loss', avg_loss)
    return avg_loss


def main(_):
    assert tfe.num_gpus() > 0, 'Make sure the GPU device exists'
    device_name = '/gpu:{}'.format(args.cuda_device)
    print('\n==> ==> ==> Using device {}'.format(device_name))

    # Load the dataset
    train_ds, val_ds = [dataset_generator(mode,
                                          conf.input_size,
                                          num_epochs=1,
                                          batch_size=conf.batch_size,
                                          buffer_size=10000)  # TODO edit this when in real training
                        for mode in ['train', 'val']]

    # Create the model and optimizer
    model = RetinaNet()
    optimizer = tf.train.RMSPropOptimizer(conf.learning_rate)

    # Define the path to the TensorBoard summary
    train_dir, val_dir = [os.path.join(conf.summary_dir, mode) for mode in ['train', 'val']]
    tf.gfile.MakeDirs(conf.summary_dir)

    train_summary_writer = tf.contrib.summary.create_summary_file_writer(train_dir, flush_millis=10000, name='train')
    val_summary_writer = tf.contrib.summary.create_summary_file_writer(val_dir, flush_millis=10000, name='val')

    checkpoint_prefix = os.path.join(conf.checkpoint_dir, 'ckpt')

    with tfe.restore_variables_on_create(tf.train.latest_checkpoint(conf.checkpoint_dir)):
        with tf.device(device_name):
            epoch = tfe.Variable(1., name='epoch')
            best_loss = tfe.Variable(tf.float32.max, name='best_loss')
            print('==> ==> ==> Start training from epoch {:.0f}...\n'.format(epoch.numpy()))

            while epoch <= conf.num_epochs + 1:
                gs = tf.train.get_or_create_global_step()
                with train_summary_writer.as_default():
                    train_one_epoch(model, optimizer, train_ds, epoch.numpy())
                with val_summary_writer.as_default():
                    eval_loss = validate(model, val_ds, epoch.numpy())

                # Save the best loss
                if eval_loss < best_loss:
                    best_loss.assign(eval_loss)  # do NOT be copied directly, SHALLOW!
                    all_variables = (model.variables + optimizer.variables() + [gs] + [epoch] + [best_loss])
                    tfe.Saver(all_variables).save(checkpoint_prefix, global_step=gs)

                epoch.assign_add(1)


if __name__ == '__main__':
    tfe.enable_eager_execution()
    tf.app.run()
