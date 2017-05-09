# coding:utf-8
"""
usage

python train_DCGAN.py --epoch 100 --ver 2
"""
import tensorflow as tf
import numpy as np
import os
import pprint
from DCGAN import *

flags = tf.app.flags
flags.DEFINE_integer("ver", 1, "Epoch to train [1]")
flags.DEFINE_integer("epoch", 50, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 64, "The size of image to use (will be center cropped). [28]")
flags.DEFINE_integer("input_width", 64, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", 64, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("retrain", True, "True for retraining, False for continuing to train [True]")
# flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
# flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()


def main(_):
    FLAGS.sample_dir = FLAGS.dataset+'_v{}_{}/'.format(FLAGS.ver,FLAGS.epoch)+FLAGS.sample_dir
    FLAGS.checkpoint_dir = FLAGS.dataset+'_v{}_{}/'.format(FLAGS.ver,FLAGS.epoch)+FLAGS.checkpoint_dir
    pp.pprint(flags.FLAGS.__flags)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    # run_config.gpu_options.allow_growth = True
    run_config.gpu_options.per_process_gpu_memory_fraction = 0.3
    with tf.Session(config=run_config) as sess:
        dcgan_model = DCGAN(sess,
                            sample_num=FLAGS.batch_size,
                            batch_size=FLAGS.batch_size,
                            input_h=FLAGS.input_height,
                            input_w=FLAGS.input_width,
                            output_h=FLAGS.output_height,
                            output_w=FLAGS.output_width,
                            c_dim=FLAGS.c_dim,
                            z_dim=100,
                            y_dim=None,
                            momentum=0.9, eps=1e-5,
                            stddev=0.02, leak=0.2,
                            input_fname_pattern=FLAGS.input_fname_pattern,
                            checkpoint_dir=FLAGS.checkpoint_dir,
                            sample_dir=FLAGS.sample_dir,
                            dataset_name=FLAGS.dataset,
                            retrain=FLAGS.retrain
                            )

    show_all_variables()
    if FLAGS.is_train:
        dcgan_model.train(FLAGS)
    else:
        if not dcgan_model.load(FLAGS.checkpoint_dir):
            raise Exception("[!] Train a model first, then run test mode")

if __name__ == '__main__':
    tf.app.run()



