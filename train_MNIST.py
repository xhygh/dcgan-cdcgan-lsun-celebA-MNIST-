# coding:utf-8

import tensorflow as tf
import numpy as np
import os
import pprint
from CDCGAN_MNIST import *

flags = tf.app.flags
flags.DEFINE_integer("ver", 2, "Epoch to train [25]")
flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_size", 28, "The size of image to use (will be center cropped). [28]")
flags.DEFINE_integer("output_height", 28, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", 28, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [mnist]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("retrain", True, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()


def main(_):
    FLAGS.sample_dir = FLAGS.dataset+'_v%d/'%FLAGS.ver+FLAGS.sample_dir
    FLAGS.checkpoint_dir = FLAGS.dataset+'_v%d/'%FLAGS.ver+FLAGS.checkpoint_dir
    pp.pprint(flags.FLAGS.__flags)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        dcgan_model = CDCGAN_MNIST(sess,
                                   sample_num=FLAGS.batch_size,
                                   batch_size=FLAGS.batch_size,
                                   img_size=FLAGS.input_size,
                                   c_dim=FLAGS.c_dim,
                                   z_dim=100,
                                   y_dim=10,
                                   filters=[1024, 128, 64],
                                   momentum=0.9, eps=1e-5,
                                   stddev=0.02, leak=0.2,
                                   checkpoint_dir=FLAGS.checkpoint_dir,
                                   sample_dir=FLAGS.sample_dir,
                                   dataset_name=FLAGS.dataset,
                                   version='v%d'%FLAGS.ver,
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



