# coding:utf-8
"""

"""
import os
import time
import tensorflow as tf
import numpy as np
from glob import glob

from DCGANops import *

image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter


class Generator(object):
    def __init__(self, inputs_dim=100,
                 nb_filters=[1024, 512, 256, 128],
                 batch_size=64,
                 output_h=64,output_w=64,
                 kenel_h=5, kenel_w=5,
                 strides_h=2, strides_w=2,
                 c_dim=3,
                 momentum=0.9,eps=1e-5,stddev=0.02):
        self.inputs_dim = inputs_dim
        self.nb_filters = nb_filters+[c_dim]
        self.batch_size = batch_size
        self.output_h = output_h
        self.output_w = output_w
        self.k_h = kenel_h
        self.k_w = kenel_w
        self.d_h = strides_h
        self.d_w = strides_w
        self.momentum = momentum
        self.eps = eps
        self.stddev = stddev

        self.g_bn0 = batch_norm(epsilon=self.eps, momentum=self.momentum, name='g_h0_bn')
        self.g_bn1 = batch_norm(epsilon=self.eps, momentum=self.momentum, name='g_h1_bn')
        self.g_bn2 = batch_norm(epsilon=self.eps, momentum=self.momentum, name='g_h2_bn')
        self.g_bn3 = batch_norm(epsilon=self.eps, momentum=self.momentum, name='g_h3_bn')

    def __call__(self, inputs, reuse=False, training=False,name=''):
        # Inputs:[N,100] 均匀分布随机数
        # Output:[N,64,64,1/3]
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()
            s_h, s_w = self.output_h, self.output_w  # G shu chu size 64
            s_h2, s_w2 = s_h/self.d_h, s_w/self.d_w  # return s_h/stride
            s_h4, s_w4 = s_h2/self.d_h, s_w2/self.d_w  # 16
            s_h8, s_w8 = s_h4/self.d_h, s_w4/self.d_w  # 8
            s_h16, s_w16 = s_h8/self.d_h, s_w8/self.d_w  # 4

            inputs = tf.convert_to_tensor(value=inputs,name='g_inputs')

            # reshape from inputs
            with tf.variable_scope('g_layer0_Reshape'):
                h0 = tf.layers.dense(inputs=inputs,
                                     units=self.nb_filters[0]*s_h16*s_w16,
                                     kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                     name='g_h0_dense')
                h0 = tf.reshape(tensor=h0,shape=[self.batch_size,s_h16,s_h16,self.nb_filters[0]],name='g_h0_reshape')
                # h0 = tf.layers.batch_normalization(inputs=h0, axis=-1, momentum=self.momentum, epsilon=self.eps,
                #                                    training=training, name='g_h0_bn')
                h0 = self.g_bn0(inputs=h0,training=training)
                h0 = tf.nn.relu(features=h0,name='g_h0_relu')
            # transpose of convolution * 4
            with tf.variable_scope('g_layer1_Conv_trans'):
                h1 = tf.layers.conv2d_transpose(inputs=h0,filters=self.nb_filters[1],
                                                kernel_size=[self.k_h,self.k_w],
                                                strides=[self.d_h,self.d_w],
                                                padding='SAME',
                                                kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                                name='g_h1_convtrans')
                # h1 = tf.layers.batch_normalization(inputs=h1,axis=-1,momentum=self.momentum,epsilon=self.eps,training=training,name='g_h1_bn')
                h1 = self.g_bn1(h1,training=training)
                h1 = tf.nn.relu(features=h1,name='g_h1_relu')

            with tf.variable_scope('g_layer2_Conv_trans'):
                h2 = tf.layers.conv2d_transpose(h1,self.nb_filters[2],
                                                [self.k_h,self.k_w],
                                                strides=[self.d_h,self.d_w],
                                                padding='SAME',
                                                kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                                name='g_h2_convtrans')
                # h2 = tf.layers.batch_normalization(h2,training=training,name='g_h2_bn')
                h2 = self.g_bn2(inputs=h2, training=training)
                h2 = tf.nn.relu(h2,name='g_h2_relu')

            with tf.variable_scope('g_layer3_Conv_trans'):
                h3 = tf.layers.conv2d_transpose(h2,self.nb_filters[3],
                                                [self.k_h,self.k_w],
                                                strides=[self.d_h,self.d_w],
                                                padding='SAME',
                                                kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                                name='g_h3_convtrans')
                # h3 = tf.layers.batch_normalization(h3,training=training,name='g_h3_bn')
                h3 = self.g_bn3(inputs=h3, training=training)
                h3 = tf.nn.relu(h3,name='g_h3_relu')

            with tf.variable_scope('g_layer4_Conv_trans'):
                h4 = tf.layers.conv2d_transpose(h3,self.nb_filters[4],
                                                [self.k_h,self.k_w],
                                                strides=[self.d_h,self.d_w],
                                                padding='SAME',
                                                kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                                name='g_h4_convtrans')

            # output layer
            with tf.variable_scope('g_layer5_Tanh_output'):
                g_output = tf.nn.tanh(h4,name='g_h5_output_tanh')
        # self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator')
        return g_output


class Discriminator(object):
    def __init__(self,inputs_h=64,inputs_w=64,
                 nb_filters=[64, 128, 256, 512],
                 batch_size=64,
                 df_h=64,df_w=64,
                 k_h=5, k_w=5,
                 d_h=2, d_w=2,
                 c_dim=3,
                 momentum=0.99,eps=1e-3,
                 stddev=0.02,leak=0.2):
        self.inputs_h = inputs_h,
        self.inputs_w = inputs_w,
        self.nb_filters = [c_dim]+nb_filters
        self.batch_size = batch_size
        self.df_h = df_h
        self.df_w = df_w
        self.k_h = k_h
        self.k_w = k_w
        self.d_h = d_h
        self.d_w = d_w
        self.momentum = momentum
        self.eps = eps
        self.stddev = stddev
        self.leak = leak

        self.d_bn0 = batch_norm(epsilon=self.eps, momentum=self.momentum, name='d_h0_bn')
        self.d_bn1 = batch_norm(epsilon=self.eps, momentum=self.momentum, name='d_h1_bn')
        self.d_bn2 = batch_norm(epsilon=self.eps, momentum=self.momentum, name='d_h2_bn')
        self.d_bn3 = batch_norm(epsilon=self.eps, momentum=self.momentum, name='d_h3_bn')

    def __call__(self, inputs, reuse=False, training=False,name=''):
        with tf.name_scope('discriminator_'+name),tf.variable_scope('discriminator',reuse=reuse):
            inputs = tf.convert_to_tensor(inputs,name='d_inputs')  # [N,inputs_h,inputs_h,1/3]

            # conv layers * 4
            with tf.variable_scope('d_layer0_Conv'):
                h0 = tf.layers.conv2d(inputs=inputs,
                                      filters=self.nb_filters[1],
                                      kernel_size=(self.k_h,self.k_w),
                                      strides=(self.d_h,self.d_w),
                                      padding='SAME',
                                      kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                      name='d_h0_conv')
                # h0 = tf.layers.batch_normalization(h0,momentum=self.momentum,epsilon=self.eps,training=training,name='d_h0_bn')
                h0 = self.d_bn0(h0,training=training)
                h0 = leakyrelu(h0,leak=self.leak,name='d_h0_lrelu')

            with tf.variable_scope('d_layer1_Conv'):
                h1 = tf.layers.conv2d(inputs=h0,
                                      filters=self.nb_filters[2],
                                      kernel_size=(self.k_h,self.k_w),
                                      strides=(self.d_h,self.d_w),
                                      padding='SAME',
                                      kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                      name='d_h1_conv')
                # h1 = tf.layers.batch_normalization(h1,momentum=self.momentum,epsilon=self.eps,training=training,name='d_h1_bn')
                h1 = self.d_bn1(h1, training=training)
                h1 = leakyrelu(h1,leak=self.leak,name='d_h1_lrelu')

            with tf.variable_scope('d_layer2_Conv'):
                h2 = tf.layers.conv2d(inputs=h1,
                                      filters=self.nb_filters[3],
                                      kernel_size=(self.k_h,self.k_w),
                                      strides=(self.d_h,self.d_w),
                                      padding='SAME',
                                      kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                      name='d_h2_conv')
                # h2 = tf.layers.batch_normalization(h2,momentum=self.momentum,epsilon=self.eps,training=training,name='d_h2_bn')
                h2 = self.d_bn2(h2, training=training)
                h2 = leakyrelu(h2,leak=self.leak,name='d_h2_lrelu')

            with tf.variable_scope('d_layer3_Conv'):
                h3 = tf.layers.conv2d(inputs=h2,
                                      filters=self.nb_filters[4],
                                      kernel_size=(self.k_h,self.k_w),
                                      strides=(self.d_h,self.d_w),
                                      padding='SAME',
                                      kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                      name='d_h3_conv')
                # h3 = tf.layers.batch_normalization(h3,momentum=self.momentum,epsilon=self.eps,training=training,name='d_h3_bn')
                h3 = self.d_bn3(h3, training=training)
                h3 = leakyrelu(h3,leak=self.leak,name='d_h3_lrelu')

            with tf.variable_scope('d_layer4_Dense'):
                h4 = tf.reshape(h3,[self.batch_size,-1],name='d_h4_reshape')  # [n,4*4*512]
                h4 = tf.layers.dense(h4,1,name='d_h4_dense')  # [n,1]

            with tf.variable_scope('d_layer5_Sigmoid'):
                d_output = tf.nn.sigmoid(h4,name='d_h5_sigmoid')  # [1,1]

        # self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return d_output, h4  # 求loss的时候会sigmoid


class DCGAN(object):
    def __init__(self,
                 sess,
                 batch_size=128,
                 sample_num=10,
                 input_h=64,input_w=64,
                 output_h=64,output_w=4,
                 z_dim=100,
                 g_filters=[1024,512,256,128],
                 d_filters=[64,128,256,512],
                 c_dim=3,
                 y_dim=None,
                 momentum=0.99,eps=1e-3,
                 stddev=0.02,leak=0.2,
                 input_fname_pattern='*.jpg',
                 retrain=True,
                 checkpoint_dir='',
                 sample_dir='',
                 dataset_name=''
                 ):
        self.sess = sess
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.input_h, self.input_w = input_h, input_w
        self.output_h = output_h
        self.output_w = output_w
        self.z_dim = z_dim

        self.g_filters = g_filters
        self.d_filters = d_filters
        self.c_dim = c_dim
        self.y_dim = y_dim
        # for batch_norm
        self.momentum = momentum
        self.eps = eps
        self.stddev = stddev
        self.leak = leak  # for leakyrelu

        self.input_fname_pattern = input_fname_pattern
        self.retrain = retrain
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.dataset_name = dataset_name

        self.generator = Generator(self.z_dim,self.g_filters,batch_size=self.batch_size,output_h=self.output_h,output_w=self.output_w,c_dim=self.c_dim,momentum=self.momentum,eps=self.eps)
        self.discriminator = Discriminator(nb_filters=self.d_filters,batch_size=self.batch_size,c_dim=self.c_dim,momentum=self.momentum,eps=self.eps)
        self.z = tf.random_uniform([self.batch_size,self.z_dim],minval=-1.,maxval=1.,name='g_z')
        self.model()
    def model(self):
        """

        :return:
        """
        # input variables
        image_dims = [self.output_h, self.output_w, self.c_dim]  #[64,64,3]
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')   # for real D
        inputs = self.inputs  #[n,64,64,3]

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')   # for G[n,100]

        # build model
        G = self.generator(self.z)  #[n,64,64,3]
        D, D_score = self.discriminator(inputs,reuse=False,training=True, name='D')  # real data
        self.sampler_ = self.sampler(self.z,reuse=True,training=False)
        D_, D_score_ = self.discriminator(G,reuse=True,training=True,name='D_')  # fake data

        # summary of input & outputs
        self.z_sum = histogram_summary("z", self.z)
        self.D_sum = histogram_summary('D',D)
        self.D__sum = histogram_summary('D_',D_)
        self.G_sum = histogram_summary('G',G)

        # Loss of G & D
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D),logits=D_score))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_),logits=D_score_))

        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_),logits=D_score_))

        # summary of Loss
        self.d_loss_real_sum = scalar_summary('d_loss_real',self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary('d_loss_fake',self.d_loss_fake)
        self.d_loss_sum = scalar_summary('d_loss',self.d_loss)
        self.g_loss_sum = scalar_summary('g_loss',self.g_loss)

        # tf.add_to_collection('g_losses',self.g_loss)
        # tf.add_to_collection('d_losses_fake', self.d_loss_fake)
        # tf.add_to_collection('d_losses_real', self.d_loss_real)
        t_vars = tf.trainable_variables()  # 返回模型中所有可训练的参数
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self,config):
        """
        :return:
        """
        # data for training
        if os.path.isdir('/home/xuhaiyue/ssd/data'):
            data = glob(os.path.join('/home/xuhaiyue/ssd/data', config.dataset, self.input_fname_pattern))  # 类似os.listdir
        else:
            data = glob(os.path.join("/media/hy/source/workspace/data/private", config.dataset, self.input_fname_pattern))  # 类似os.listdir

        # g_optim = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1).minimize(self.g_loss,self.generator.variables)
        # d_optim = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1).minimize(self.d_loss,self.discriminator.variables)
        g_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate,beta1=config.beta1).minimize(self.g_loss,var_list=self.g_vars)
        d_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate,beta1=config.beta1).minimize(self.d_loss,var_list=self.d_vars)

        # 初始化变量
        self.sess.run(tf.global_variables_initializer())
        # 合并各自的总结
        self.g_sum = merge_summary([self.z_sum, self.D__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary([self.z_sum, self.D_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter('./{}_v{}_{}/logs'.format(config.dataset,config.ver,config.epoch), self.sess.graph)

        # 采样测试网络的数据
        sample_dataset = data[0:self.sample_num]  # 以开始的self.sample_num个训练图像作为采样
        sample = [get_img(sample_data,self.input_h,self.input_w,self.output_h,self.output_w,self.c_dim) for sample_data in sample_dataset]
        sample_inputs = np.array(sample).astype(np.float32)
        # 用于采样器，生成图像的z，采样器相当于对于G效果的验证
        sample_z = np.random.uniform(-1,1,(self.sample_num,self.z_dim)).astype(np.float32)  #[sample_num, z_dim]

        step = 1
        start_time = time.time()
        #######################
        # 需要看一下
        if not self.retrain:
            could_load, checkpoint_step = self.load(self.checkpoint_dir)
            if could_load:
                step = checkpoint_step
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        #########################
        ##########################
        # Training stage
        ##########################
        print ('************Start training*************')
        for epoch in range(config.epoch):
            # 分成batch_size大小的几块
            batch_idxs = min(len(data), config.train_size) // config.batch_size

            # for each batch
            for idx in range(batch_idxs):
                # prepare data for G & D
                batch_dataset = data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_img(batch_data,self.input_h,self.input_w,self.output_h,self.output_w,self.c_dim) for batch_data in batch_dataset]
                batch_inputs = np.array(batch).astype(np.float32)
                batch_z = np.random.uniform(-1,1,(self.batch_size,self.z_dim)).astype(np.float32)  # [batch_size, z_dim]
                ###############
                # Training
                # first, update D net once
                # second, update G net twice
                # thirdly, repeat the above steps
                ###############
                # update D
                _,summary_str = self.sess.run([d_optim,self.d_sum],
                                              feed_dict={self.inputs:batch_inputs,
                                                         self.z:batch_z})
                self.writer.add_summary(summary_str,step)

                # first update G
                _,summary_str = self.sess.run([g_optim,self.g_sum],
                                              feed_dict={self.z:batch_z})
                self.writer.add_summary(summary_str, step)
                # second update G
                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _,summary_str = self.sess.run([g_optim,self.g_sum],
                                              feed_dict={self.z:batch_z})
                self.writer.add_summary(summary_str, step)

                # record loss
                err_d_fake_loss = self.d_loss_fake.eval(feed_dict={self.z:batch_z},session=self.sess)
                err_d_real_loss = self.d_loss_real.eval(feed_dict={self.inputs:batch_inputs},session=self.sess)
                err_d = self.d_loss.eval(feed_dict={self.z: batch_z, self.inputs: batch_inputs},session=self.sess)
                err_g = self.g_loss.eval(feed_dict={self.z:batch_z},session=self.sess)



                # print result
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_fake_loss: %.8f, d_real_loss: %.8f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time,
                         err_d_fake_loss,err_d_real_loss,
                         err_d, err_g))

                # sample validate
                if np.mod(step, 100) == 0:
                    try:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler_, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                            },
                        )
                        manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                        manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                        save_images(samples, [manifold_h, manifold_w],
                                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    except:
                        print("one pic error!...")

                if np.mod(step, 500) == 0:
                    self.save(config.checkpoint_dir, step)  # 关于global_step可以看下官网

                # 本次batch更新结束
                step += 1
            print('*************Finished epoch %d***********' % epoch)

    def sampler(self, z, y=None,reuse=True,training=False):
        """
        训练过程中用于采样显示G网络效果的函数
        :param z: [N,100]
        :param y: None or [N,self.y_dim]
        :param reuse: True，复用训练好的G网络的参数
        :param training: False,测试G性能，不需要训练
        :return:
        """
        return self.generator(z,reuse=reuse,training=training)

    def test(self,config):
        # data for training & test
        if os.path.isdir('/home/xuhaiyue/ssd/data'):
            data = glob(os.path.join('/home/xuhaiyue/ssd/data', config.dataset, self.input_fname_pattern))  # 类似os.listdir
        else:
            data = glob(os.path.join("/media/hy/source/workspace/data/private", config.dataset, self.input_fname_pattern))  # 类似os.listdir

        test_dataset = data[0:self.sample_num]  # 以开始的self.sample_num个训练图像作为采样
        sample = [get_img(sample_data, self.input_h, self.input_w, self.output_h, self.output_w, self.c_dim) for
                  sample_data in test_dataset]
        sample_inputs = np.array(sample).astype(np.float32)
        # 用于采样器，生成图像的z，采样器相当于对于G效果的验证
        sample_z = np.random.uniform(-1, 1, (self.sample_num, self.z_dim)).astype(np.float32)  # [sample_num, z_dim]
        try:
            samples, d_loss, g_loss = self.sess.run(
                [self.sampler_, self.d_loss, self.g_loss],
                feed_dict={
                    self.z: sample_z,
                    self.inputs: sample_inputs,
                },
            )
            manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
            manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
            save_images(samples, [manifold_h, manifold_w],
                        './{}/test.png'.format(config.sample_dir))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
        except:
            print("one pic error!...")


    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_h, self.output_w)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            step = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, step
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
