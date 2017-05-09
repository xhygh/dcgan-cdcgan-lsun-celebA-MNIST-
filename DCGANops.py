# coding:utf-8

import tensorflow as tf
import numpy as np
import scipy.misc
import tensorflow.contrib.slim as slim


def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def inverse_transform(images):
  return (images+1.)/2.


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)


def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)



def leakyrelu(x,leak=0.2,name='lrelu'):
    return tf.maximum(x,leak*x,name=name)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias



class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
          self.epsilon = epsilon
          self.momentum = momentum
          self.name = name

    def __call__(self, inputs, training=True):
        return tf.contrib.layers.batch_norm(inputs,
                          decay=self.momentum,
                          updates_collections=None,
                          epsilon=self.epsilon,
                          scale=True,
                          is_training=training,
                          scope=self.name)

def get_img(img_path,input_h,input_w,output_h,output_w,c_dim):
    """
    读入一个image，resize到指定size
    :param img_path:一个image的绝对路径
    :param input_h:
    :param input_w:
    :param output_h:
    :param output_w:
    :param c_dim:
    :return:
    """

    img = scipy.misc.imread(img_path).astype(np.float)  # read a img
    resize_img = scipy.misc.imresize(img,[output_h,output_w])  # resize 输入图像
    norm = np.array(resize_img)/127.5 - 1.  # 归一化到-1～1
    return norm  # [1,h,w,c]

