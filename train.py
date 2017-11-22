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
parser.add_argument('--resume', '-r', default='store_true', help='resume from checkpoint')
parser.add_argument('--cuda-device', default=0, type=int, help='gpu device index')

args = parser.parse_args()


def train_one_epoch(model, optimizer, dataset, log_interval=None):
    """Trains model on `dataset` using `optimizer`."""

    tf.train.get_or_create_global_step()

    def model_loss(images, loc_trues, cls_trues):
        loc_preds, cls_preds = model(images, training=True)
        loss_value = loss_fn(loc_preds, loc_trues, cls_preds, cls_trues, num_classes=conf.num_class)
        tf.contrib.summary.scalar('loss', loss_value)
        return loss_value

    for batch, (images, loc_trues, cls_trues) in enumerate(tfe.Iterator(dataset)):
        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            gs = tf.train.get_global_step()
            # Visualize the input images
            tf.contrib.summary.image('inputs', tf.transpose(images, [0, 2, 3, 1]), max_images=2, global_step=gs)
            # Optimize the loss
            batch_model_loss = functools.partial(model_loss, images, loc_trues, cls_trues)
            optimizer.minimize(batch_model_loss, global_step=gs)
            if log_interval and batch % log_interval == 0:
                print('==> ==> ==> Batch #{}\tLoss: {:.6f}'.format(batch, batch_model_loss().numpy()))


def validate(model, dataset):
    """Perform an evaluation of `model` on the examples from `dataset`."""

    total_loss = 0.
    for batch, (images, loc_trues, cls_trues) in enumerate(tfe.Iterator(dataset)):
        loc_preds, cls_preds = model(images)
        loss = loss_fn(loc_preds, loc_trues, cls_preds, cls_trues, num_classes=conf.num_class)
        total_loss += loss
        if batch % 10 == 0:
            print('==> ==> ==> Batch #{}\t Test loss: {:.6f}'.format(batch, loss.numpy()))

    avg_loss = (total_loss / batch).numpy()
    print('Test set: Average loss: {:.6f}\n'.format(avg_loss))
    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('loss', avg_loss)


def main(_):
    assert tfe.num_gpus() > 0, 'Make sure the GPU device exists'
    device_name = '/gpu:{}'.format(args.cuda_device)
    print('==> ==> ==> Using device {}'.format(device_name))

    # Load the dataset
    train_ds, val_ds = [dataset_generator(mode,
                                          conf.input_size,
                                          num_epochs=1,
                                          batch_size=conf.batch_size,
                                          buffer_size=100)
                        # buffer_size=conf.buffer_size)
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

    with tf.device(device_name):
        for epoch in range(conf.num_epochs):
            with tfe.restore_variables_on_create(tf.train.latest_checkpoint(conf.checkpoint_dir)):
                global_step = tf.train.get_or_create_global_step()
                start = time.time()
                with train_summary_writer.as_default():
                    train_one_epoch(model, optimizer, train_ds, conf.log_interval)
                end = time.time()
                print('==> ==> ==> Train time for epoch #{} (global step {}): {:.3f}s\n'.format(
                        epoch, global_step.numpy(), end - start))
                with val_summary_writer.as_default():
                    validate(model, val_ds)
                all_variables = (model.variables + optimizer.variables() + [global_step])
                tfe.Saver(all_variables).save(checkpoint_prefix, global_step=global_step)


if __name__ == '__main__':
    tfe.enable_eager_execution()
    tf.app.run()
