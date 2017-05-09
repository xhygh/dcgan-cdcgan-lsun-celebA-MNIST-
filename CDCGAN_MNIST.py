# coding:utf-8
"""
https://github.com/sugyan/tf-dcgan

添加记录部分
"""
import os
import time


from DCGANops import *

image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter


class CDCGAN_MNIST(object):
    """
    generator MNIST digits with C-dcgan
    """
    def __init__(self, sess, sample_num=10,
                 batch_size=128,img_size=28,
                 c_dim=1,z_dim=100,y_dim=10,
                 filters=[1024,128,64],
                 momentum=0.99,eps=1e-5,
                 stddev=0.02,leak=0.2,
                 checkpoint_dir='',
                 sample_dir='',
                 dataset_name='',
                 version='v3',
                 retrain=False):
        self.sess = sess
        self.sample_num = sample_num
        self.batch_size = batch_size
        self.img_size = img_size
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.filters = filters
        self.momentum = momentum
        self.eps = eps
        self.stddev = stddev
        self.leak = leak
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.dataset_name = dataset_name
        self.version = version
        self.retrain = retrain

        self.d_bn0 = batch_norm(name='d_bn0')  # v1 & v4
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        self.build_model()




    def build_model(self,training=True):
        self.y = tf.placeholder(tf.float32,[self.batch_size,self.y_dim],name='y')
        img_dims = [self.img_size,self.img_size,self.c_dim]
        self.inputs = tf.placeholder(tf.float32,[self.batch_size]+img_dims,name='real_imgs')  #[N,28,28,1]

        inputs = self.inputs


        self.z = tf.placeholder(tf.float32,[None,self.z_dim],name='z')
        self.z_sum = histogram_summary('z',self.z)  # 总结
        if self.version == 'v1':
            print('version1 GAN')
            print(
                "使用tf.layers.bn 和tf.layers.conv,tf.layers.conv_trans\n"
                "CG c_dim:[100/110,d-1024/1034, d-128/138, c-64/74, c-1]\n"
                "CD c_dim:[1/11, c-64/74,c-128/7*7*128+10, d-1024/1034,d-1]\n"
            )
            self.G = self.generator(self.z,self.y,training=training)  # 生成图像
            self.D, self.D_log = self.discriminator(inputs,self.y,reuse=False,training=training)  # 判别真是图像
            self.sampler = self.sampler(self.z, self.y)  # 采样图像，方便观察
            self.D_, self.D_log_ = self.discriminator(self.G,self.y,reuse=True,training=training)  # 判别生成图像
        elif self.version == 'v2':
            print('version2 GAN')
            print(
                "使用tf.layers.bn 和tf.layers.conv,tf.layers.conv_trans\n"
                "CG c_dim:[100/110,d-1024/1034, d-128*7*7/128/138, c-128/138, c-1]\n"
                "CD c_dim:[1/11, c-11/21,c-74/7*7*74+10, d-1024/1034,d-1]\n"
            )
            self.G = self.generator_v2(self.z,self.y,training=training)  # 生成图像
            self.D, self.D_log = self.discriminator_v2(inputs,self.y,reuse=False,training=training)  # 判别真是图像
            self.sampler = self.sampler_v2(self.z, self.y)  # 采样图像，方便观察
            self.D_, self.D_log_ = self.discriminator_v2(self.G,self.y,reuse=True,training=training)  # 判别生成图像

        elif self.version == 'v3':
            print('version3 GAN')
            """修改了BN层，BN层使用contib提供的，超参相同"""
            "CG c_dim:[100/110,d-1024/1034, d-128*7*7/128/138, c-128/138, c-1]\n"
            "CD c_dim:[1/11, c-11/21,c-74/7*7*74+10, d-1024/1034,d-1]\n"
            self.G = self.generator_v3(self.z,self.y,training=training)  # 生成图像
            self.D, self.D_log = self.discriminator_v3(inputs,self.y,reuse=False,training=training)  # 判别真是图像
            self.sampler = self.sampler_v3(self.z, self.y)  # 采样图像，方便观察
            self.D_, self.D_log_ = self.discriminator_v3(self.G,self.y,reuse=True,training=training)  # 判别生成图像
        else:
            print('version4 GAN')
            """修改了BN层，BN层使用contib提供的，超参相同"""
            "CG c_dim:[100/110,d-1024/1034, d-128/138, c-64/74, c-1]\n"
            "CD c_dim:[1/11, c-64/74,c-128/7*7*128+10, d-1024/1034,d-1]\n"
            self.G = self.generator_v3(self.z,self.y,training=training)  # 生成图像
            self.D, self.D_log = self.discriminator_v3(inputs,self.y,reuse=False,training=training)  # 判别真是图像
            self.sampler = self.sampler_v3(self.z, self.y)  # 采样图像，方便观察
            self.D_, self.D_log_ = self.discriminator_v3(self.G,self.y,reuse=True,training=training)  # 判别生成图像

        self.d_sum = histogram_summary("d", self.D)  # 总结
        self.d__sum = histogram_summary("d_", self.D_)  # 总结
        self.G_sum = image_summary("G", self.G)  # 总结



        # 计算loss
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D),logits=self.D_log))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_),logits=self.D_log_))

        self.d_loss = self.d_loss_fake + self.d_loss_real

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_),logits=self.D_log_))  # 假数据判真的宋史

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)# 总结
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)# 总结

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)# 总结
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)# 总结

        # G和D需要更新的变量，用于优化器中的var_list
        t_vars = tf.trainable_variables()  # 返回模型中所有可训练的参数

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    def train(self,config):
        print ("Loading MNIST data...")
        data_X, data_y = self.load_mnist()
        print ("Finished loading MNIST data...")

        d_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate,beta1=config.beta1).minimize(self.d_loss,var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate,beta1=config.beta1).minimize(self.g_loss,var_list=self.g_vars)

        # 初始化变量
        self.sess.run(tf.global_variables_initializer())
        # 合并各自的总结
        self.g_sum = merge_summary([self.z_sum,self.d__sum,self.G_sum,self.d_loss_fake_sum,self.g_loss_sum])
        self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter('./%s_v%d/logs'%(config.dataset,config.ver),self.sess.graph)

        #用于采样器，生成图像的z，采样器相当于对于G效果的验证
        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
        # 以最开始的batch_size数量训练数据做对比
        sample_inputs = data_X[0:self.sample_num]
        sample_labels = data_y[0:self.sample_num]

        counter = 1
        start_time = time.time()
        #######################
        # 需要看一下
        if not self.retrain:
            could_load, checkpoint_counter = self.load(self.checkpoint_dir)
            if could_load:
                counter = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                #########################
        # 训练config.epoch个epoch
        for epoch in range(config.epoch):
            batch_idx = min(len(data_X), config.train_size) // config.batch_size

            for idx in range(0,batch_idx):
                # 载入每个batch的训练数据
                batch_imgs = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_labs = data_y[idx * config.batch_size:(idx + 1) * config.batch_size]
                #载入每个batch的虚假数据
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # 更新D网络
                _, summary_str = self.sess.run([d_optim,self.d_sum],
                                              feed_dict={
                                                  self.inputs:batch_imgs,
                                                  self.z:batch_z,
                                                  self.y:batch_labs
                                              })
                self.writer.add_summary(summary_str, counter)

                # 更新G网络,连续更新 g_optim 两次，保证D不会先loss=0(different from paper)
                _, summary_str = self.sess.run([g_optim,self.g_sum],
                                               feed_dict={
                                                   self.z: batch_z,
                                                   self.y: batch_labs
                                               })
                self.writer.add_summary(summary_str, counter)

                _, summary_str = self.sess.run([g_optim,self.g_sum],
                                               feed_dict={
                                                   self.z: batch_z,
                                                   self.y: batch_labs
                                               })
                self.writer.add_summary(summary_str, counter)

                # 记录D判错和判对的err
                errD_fake = self.d_loss_fake.eval({
                    self.z: batch_z,
                    self.y: batch_labs
                },session=self.sess)
                errD_real = self.d_loss_real.eval({
                    self.inputs: batch_imgs,
                    self.y: batch_labs
                },session=self.sess)
                # 记录G片过去的err
                errG = self.g_loss.eval({
                    self.z: batch_z,
                    self.y: batch_labs
                },session=self.sess)
                errD = errD_fake+errD_real
                counter += 1
                print('Epoch: [%2d] [%4d/%4d] time: %4.4f, d_fake_loss:%.8f, d_real_loss:%.8f, d_loss: %.8f, g_loss: %.8f'\
                      %(epoch,idx,batch_idx,
                        time.time()-start_time,
                        errD_fake,errD_real,errD,errG))
                # 训练过程中，采样器采样，并保存采样图片（G生成的）
                if np.mod(counter,100) ==0:

                    samples, d_loss, g_loss,d,d_ = self.sess.run([self.sampler, self.d_loss,self.g_loss,self.D,self.D_],
                                                            feed_dict={self.z:sample_z,
                                                                       self.inputs:sample_inputs,
                                                                       self.y:sample_labels,
                                                                   })
                    manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                    manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                    save_images(samples, [manifold_h, manifold_w],
                                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f, D:%.8f,D_:%.8f" % (d_loss, g_loss,np.mean(np.array(d)),np.mean(np.array(d_))))
                # 保存一次模型
                if np.mod(counter, 200) == 0:
                    self.save(config.checkpoint_dir, counter)
            print('*************Finished epoch %d***********'%epoch)

    def sampler(self,z,y,training=False):
        """
        采样生成结果的
        :param z:
        :param y:
        :return:
        """
        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()

            yb = tf.reshape(y,[self.batch_size,1,1,self.y_dim])  # [N,10]->[N,1,1,10]
            z = tf.concat(values=[z,y],axis=1)   #[N,110]

            h0 = tf.nn.relu(tf.layers.batch_normalization(linear(z,self.filters[0],'g_h0_lin'),
                                                         training=training,
                                                         name='g_h0_bn'),
                           name='g_h0_relu')  #[N,1024]
            h0 = tf.concat(values=[h0,y],axis=1)  # [N,1024+10]

            h1 = tf.nn.relu(tf.layers.batch_normalization(linear(h0,self.filters[1]*7*7,'g_h1_lin'),
                                                          training=training,
                                                          name='g_h1_bn'),
                            name='g_h1_relu')  #[N,128*7*7]
            h1 = tf.reshape(h1,[self.batch_size,7,7,self.filters[1]])  #[N,7,7,128]
            h1 = tf.concat(axis=3,values=[h1,yb*tf.ones([self.batch_size,7,7,self.y_dim])])  #[N,7,7,128+10]

            h2 = tf.layers.conv2d_transpose(h1,self.filters[2],5,(2,2),'SAME',
                                            kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),name='g_h2_convtrans')
            h2 = tf.nn.relu(tf.layers.batch_normalization(h2,training=training,name='g_h2_bn'),name='g_h2_relu')  #[N,14,14,64]
            h2 = tf.concat(values=[h2,yb*tf.ones([self.batch_size,14,14,self.y_dim])],axis=3)  #[n,14,14,64+10]

            h3 = tf.layers.conv2d_transpose(h2,self.c_dim,5,(2,2),'SAME',kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),name='g_h3_convtrans')

            h4 = tf.nn.sigmoid(h3,name='g_h4_sig')
            return h4
    def sampler_v2(self,z,y,training=False):
        """
        采样生成结果的
        :param z:
        :param y:
        :return:
        """

        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()

            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])  # n,1,1,10
            z = tf.concat([z, y], 1)  # 64, 110

            h0 = tf.nn.relu(
                tf.layers.batch_normalization(linear(z, 1024, 'g_h0_lin', stddev=self.stddev), training=training,
                                              name='g_h0_bn'), name='g_h0_relu')  # n,1024
            h0 = tf.concat([h0, y], 1)  # [64,1034]

            h1 = tf.nn.relu(tf.layers.batch_normalization(
                linear(h0, 64 * 2 * 7 * 7, 'g_h1_lin', stddev=self.stddev), training=training, name='g_h1_bn'),
                name='g_h1_relu')  # [64,64*2*7*7]
            h1 = tf.reshape(h1, [self.batch_size, 7, 7, 64 * 2])  # [64,7,7,64*2]
            h1 = tf.concat(axis=3,
                           values=[h1, yb * tf.ones([self.batch_size, 7, 7, self.y_dim])])  # [64,7,7,64*2+10]

            h2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(h1, 64 * 2, 5, (2, 2), 'SAME',
                                                                                     kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                                                                     name='g_h2_convtrans'),
                                                          training=training, name='g_h2_bn')
                            , name='g_h2_relu')  # [64,14,14,64*2]
            h2 = tf.concat(axis=3,
                           values=[h2, yb * tf.ones([self.batch_size, 14, 14, self.y_dim])])  # [64,14,14,64*2+10]

            return tf.nn.sigmoid(
                tf.layers.conv2d_transpose(h2, 1, 5, (2, 2), 'SAME',
                                           kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                           name='g_h3_convtrans'))  # n, 28, 28, 1

    def sampler_v3(self,z,y,training=False):
        """
        采样生成结果的
        :param z:
        :param y:
        :return:
        """

        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()
            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])  # n,1,1,10
            z = tf.concat([z, y], 1)  # 64, 110

            h0 = tf.nn.relu(
                self.g_bn0(linear(z, 1024, 'g_h0_lin', stddev=self.stddev), training=training), name='g_h0_relu')  # n,1024
            h0 = tf.concat([h0, y], 1)  # [64,1034]

            h1 = tf.nn.relu(self.g_bn1(
                linear(h0, 64 * 2 * 7 * 7, 'g_h1_lin', stddev=self.stddev), training=training),
                name='g_h1_relu')  # [64,64*2*7*7]
            h1 = tf.reshape(h1, [self.batch_size, 7, 7, 64 * 2])  # [64,7,7,64*2]
            h1 = tf.concat(axis=3,
                           values=[h1, yb * tf.ones([self.batch_size, 7, 7, self.y_dim])])  # [64,7,7,64*2+10]

            h2 = tf.nn.relu(self.g_bn2(tf.layers.conv2d_transpose(h1, 64 * 2, 5, (2, 2), 'SAME',
                                                                                     kernel_initializer=tf.random_normal_initializer(
                                                                                         stddev=self.stddev),
                                                                                     name='g_h2_convtrans'), training=training),
                            name='g_h2_relu')  # [64,14,14,64*2]
            h2 = tf.concat(axis=3,
                           values=[h2, yb * tf.ones([self.batch_size, 14, 14, self.y_dim])])  # [64,14,14,64*2+10]

            return tf.nn.sigmoid(
                tf.layers.conv2d_transpose(h2, 1, 5, (2, 2), 'SAME',
                                           kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                           name='g_h3_convtrans'))  # n, 28, 28, 1

    def sampler_v4(self,z,y,training=False):
        """
        采样生成结果的
        :param z:
        :param y:
        :return:
        """
        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()

            yb = tf.reshape(y,[self.batch_size,1,1,self.y_dim])  # [N,10]->[N,1,1,10]
            z = tf.concat(values=[z,y],axis=1)   #[N,110]

            h0 = tf.nn.relu(self.g_bn0(linear(z,self.filters[0],'g_h0_lin'),
                                                         training=training),
                           name='g_h0_relu')  #[N,1024]
            h0 = tf.concat(values=[h0,y],axis=1)  # [N,1024+10]

            h1 = tf.nn.relu(self.g_bn1(linear(h0,self.filters[1]*7*7,'g_h1_lin'),
                                                          training=training),
                            name='g_h1_relu')  #[N,128*7*7]
            h1 = tf.reshape(h1,[self.batch_size,7,7,self.filters[1]])  #[N,7,7,128]
            h1 = tf.concat(axis=3,values=[h1,yb*tf.ones([self.batch_size,7,7,self.y_dim])])  #[N,7,7,128+10]

            h2 = tf.layers.conv2d_transpose(h1,self.filters[2],5,(2,2),'SAME',
                                            kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),name='g_h2_convtrans')
            h2 = tf.nn.relu(self.g_bn2(h2,training=training),name='g_h2_relu')  #[N,14,14,64]
            h2 = tf.concat(values=[h2,yb*tf.ones([self.batch_size,14,14,self.y_dim])],axis=3)  #[n,14,14,64+10]

            h3 = tf.layers.conv2d_transpose(h2,self.c_dim,5,(2,2),'SAME',kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),name='g_h3_convtrans')

            h4 = tf.nn.sigmoid(h3,name='g_h4_sig')
            return h4


    def discriminator(self,img,y,reuse=False,training=False):
        # 真伪数据都要通过D网络，使用同一组参数，所以设置复用
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            # 串联约束条件
            yb = tf.reshape(y,[self.batch_size,1,1,self.y_dim])  # [N,10]->[N,1,1,10]
            x = tf.concat(axis=3,values=[img,yb*tf.ones([self.batch_size,self.img_size,self.img_size,self.y_dim])])  #[N,28,28,1]->[N,28,28,1+10]

            # 卷积，激活，串联条件
            h0 = leakyrelu(tf.layers.conv2d(x,self.filters[2],5,(2,2),'SAME',
                                            kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                            name='d_h0_conv'),name='d_h0_lrelu')  #[N,28,28,11] -> [N,14,14,64]
            h0 = tf.concat(axis=3,values=[h0,yb*tf.ones([self.batch_size,self.img_size/2,self.img_size/2,self.y_dim])])  # [N,14,14,64+10]

            # 卷积，BN，激活，reshape，串联
            h1 = leakyrelu(tf.layers.batch_normalization(tf.layers.conv2d(h0,self.filters[1],5,(2,2),'SAME',
                                                                          kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                                                          name='d_h1_conv'),
                                                         momentum=self.momentum,epsilon=self.eps,
                                                         training=training,name='d_h1_bn'),
                           name='d_h1_lrelu')  #[N,7,7,128]
            h1 = tf.reshape(h1,[self.batch_size,-1])  # [N,7*7*128]
            h1 = tf.concat(axis=1,values=[h1,y])  # [N,7*7*128+10]

            # 全连接，BN，激活，串联条件
            h2 = leakyrelu(tf.layers.batch_normalization(linear(h1,self.filters[0],scope='d_h2_lin',stddev=self.stddev),
                                                         training=training,name='d_h2_bn'),name='d_h2_lrelu')  # [N,7*7*128+10]->[N,1024]
            h2 = tf.concat(axis=1,values=[h2,y])  # [N,1024+10]

            # 全链接，输出score
            h3 = linear(h2,1,'d_h3_lin',stddev=self.stddev)  # [N,1]

            return tf.nn.sigmoid(h3,name='d_h3_sig'), h3

    def discriminator_v2(self, img, y,reuse=False, training=False):
        kinit = tf.truncated_normal_initializer(stddev=self.stddev)
        # kinit = tf.random_normal_initializer(stddev=self.stddev)
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])  # [N,1,1,y_dim]
            x = tf.concat(axis=3,values=[img,yb*tf.ones([self.batch_size,self.img_size,self.img_size,self.y_dim])])   # [N,28,28,1+10]

            h0 = leakyrelu(tf.layers.conv2d(x, self.c_dim + self.y_dim, 5, (2, 2), 'SAME',
                                            kernel_initializer=kinit,
                                            name='d_h0_conv'), name='d_h0_lrelu')  # [N,28,28,11] -> [N,14,14,1+10]
            h0 = tf.concat(axis=3, values=[h0, yb * tf.ones([self.batch_size, self.img_size/2, self.img_size/2, self.y_dim])]) # [N,14,14,11+10]

            h1 = leakyrelu(tf.layers.batch_normalization(tf.layers.conv2d(h0, 64 + self.y_dim, 5,(2,2),'SAME',
                                                                          kernel_initializer=kinit,
                                                                          name='d_h1_conv'),training=training,name='d_h1_bn'),name='d_h1_lrelu')  # [N,7,7,64+10]
            h1 = tf.reshape(h1, [self.batch_size, -1])  # [N,7*7*74]
            h1 = tf.concat([h1, y], 1)  # [N,7*7*74+10]

            h2 = leakyrelu(tf.layers.batch_normalization(linear(h1, 1024, 'd_h2_lin',stddev=self.stddev),training=training,name='d_h2_bn'),name='d_h2_lrelu')  # [N,1024]
            h2 = tf.concat([h2, y], 1)  # [N,1024+10]

            h3 = linear(h2, 1, 'd_h3_lin',stddev=self.stddev)  # [N,1]

            return tf.nn.sigmoid(h3,name='d_h3_sig'), h3

    def discriminator_v3(self, img, y, reuse=False, training=False):
        # kinit = tf.truncated_normal_initializer(stddev=self.stddev)
        kinit = tf.random_normal_initializer(stddev=self.stddev)

        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])  # [N,1,1,y_dim]
            x = tf.concat(axis=3, values=[img, yb * tf.ones(
                [self.batch_size, self.img_size, self.img_size, self.y_dim])])  # [N,28,28,1+10]

            h0 = leakyrelu(tf.layers.conv2d(x, self.c_dim + self.y_dim, 5, (2, 2), 'SAME',
                                            kernel_initializer=kinit,
                                            name='d_h0_conv'), name='d_h0_lrelu')  # [N,28,28,11] -> [N,14,14,1+10]
            h0 = tf.concat(axis=3, values=[h0, yb * tf.ones(
                [self.batch_size, self.img_size / 2, self.img_size / 2, self.y_dim])])  # [N,14,14,11+10]

            h1 = leakyrelu(self.d_bn1(tf.layers.conv2d(h0, 64 + self.y_dim, 5, (2, 2), 'SAME',
                                                                          kernel_initializer=kinit,
                                                                          name='d_h1_conv'), training=training),
                           name='d_h1_lrelu')  # [N,7,7,64+10]
            h1 = tf.reshape(h1, [self.batch_size, -1])  # [N,7*7*74]
            h1 = tf.concat([h1, y], 1)  # [N,7*7*74+10]

            h2 = leakyrelu(
                self.d_bn2(linear(h1, 1024, 'd_h2_lin', stddev=self.stddev), training=training),
                name='d_h2_lrelu')  # [N,1024]
            h2 = tf.concat([h2, y], 1)  # [N,1024+10]

            h3 = linear(h2, 1, 'd_h3_lin', stddev=self.stddev)  # [N,1]

            return tf.nn.sigmoid(h3, name='d_h3_sig'), h3

    def discriminator_v4(self,img,y,reuse=False,training=False):
        # 真伪数据都要通过D网络，使用同一组参数，所以设置复用
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            # 串联约束条件
            yb = tf.reshape(y,[self.batch_size,1,1,self.y_dim])  # [N,10]->[N,1,1,10]
            x = tf.concat(axis=3,values=[img,yb*tf.ones([self.batch_size,self.img_size,self.img_size,self.y_dim])])  #[N,28,28,1]->[N,28,28,1+10]

            # 卷积，激活，串联条件
            h0 = leakyrelu(tf.layers.conv2d(x,self.filters[2],5,(2,2),'SAME',
                                            kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                            name='d_h0_conv'),name='d_h0_lrelu')  #[N,28,28,11] -> [N,14,14,64]
            h0 = tf.concat(axis=3,values=[h0,yb*tf.ones([self.batch_size,self.img_size/2,self.img_size/2,self.y_dim])])  # [N,14,14,64+10]

            # 卷积，BN，激活，reshape，串联
            h1 = leakyrelu(self.d_bn1(tf.layers.conv2d(h0,self.filters[1],5,(2,2),'SAME',
                                                                          kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                                                          name='d_h1_conv'),training=training),name='d_h1_lrelu')  #[N,7,7,128]
            h1 = tf.reshape(h1,[self.batch_size,-1])  # [N,7*7*128]
            h1 = tf.concat(axis=1,values=[h1,y])  # [N,7*7*128+10]

            # 全连接，BN，激活，串联条件
            h2 = leakyrelu(self.d_bn2(linear(h1,self.filters[0],scope='d_h2_lin',stddev=self.stddev),
                                                         training=training),name='d_h2_lrelu')  # [N,7*7*128+10]->[N,1024]
            h2 = tf.concat(axis=1,values=[h2,y])  # [N,1024+10]

            # 全链接，输出score
            h3 = linear(h2,1,'d_h3_lin',stddev=self.stddev)  # [N,1]

            return tf.nn.sigmoid(h3,name='d_h3_sig'), h3

    def generator(self,z,y,training=False):
        # z [N,100]
        # y [N,10]
        with tf.variable_scope('generator') as scope:
            # label扩张维度，输入串联条件
            yb = tf.reshape(y,[self.batch_size,1,1,self.y_dim])  # [N,10]->[N,1,1,10]
            z = tf.concat(values=[z,y],axis=1)   # [N,110]

            # 全链接，BN，激活，串联条件
            h0 = tf.nn.relu(tf.layers.batch_normalization(linear(z,self.filters[0],'g_h0_lin'),
                                                         training=training,
                                                         name='g_h0_bn'),
                           name='g_h0_relu')  #[N,1024]
            h0 = tf.concat(values=[h0,y],axis=1)  # [N,1024+10]

            # 全链接，BN，激活，reshape，串联条件
            h1 = tf.nn.relu(tf.layers.batch_normalization(linear(h0,self.filters[1]*7*7,'g_h1_lin'),
                                                          training=training,
                                                          name='g_h1_bn'),
                            name='g_h1_relu')  #[N,128*7*7]
            h1 = tf.reshape(h1,[self.batch_size,7,7,self.filters[1]])  #[N,7,7,128]
            h1 = tf.concat(axis=3,values=[h1,yb*tf.ones([self.batch_size,7,7,self.y_dim])])  #[N,7,7,128+10]

            # 转置卷积，BN，激活，串联条件
            h2 = tf.layers.conv2d_transpose(h1,self.filters[2],5,(2,2),'SAME',
                                            kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),name='g_h2_convtrans')
            h2 = tf.nn.relu(tf.layers.batch_normalization(h2,training=training,name='g_h2_bn'),name='g_h2_relu')  #[N,14,14,64]
            h2 = tf.concat(values=[h2,yb*tf.ones([self.batch_size,14,14,self.y_dim])],axis=3)  #[n,14,14,64+10]

            # 转置卷积
            h3 = tf.layers.conv2d_transpose(h2,self.c_dim,5,(2,2),'SAME',kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),name='g_h3_convtrans')
            # 归一道0,1之间
            h4 = tf.nn.sigmoid(h3,name='g_h4_sig')
            return h4

    def generator_v2(self,z,y,training=False):
        with tf.variable_scope("generator") as scope:


            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])  # n,1,1,10
            z = tf.concat([z, y], 1)  # 64, 110

            h0 = tf.nn.relu(
                tf.layers.batch_normalization(linear(z, 1024, 'g_h0_lin',stddev=self.stddev),training=training,name='g_h0_bn'),name='g_h0_relu')  # n,1024
            h0 = tf.concat([h0, y], 1)  # [64,1034]

            h1 = tf.nn.relu(tf.layers.batch_normalization(
                linear(h0, 64 * 2 * 7 * 7, 'g_h1_lin',stddev=self.stddev),training=training,name='g_h1_bn'),name='g_h1_relu')  # [64,64*2*7*7]
            h1 = tf.reshape(h1, [self.batch_size, 7, 7, 64 * 2])  # [64,7,7,64*2]
            h1 = tf.concat(axis=3, values=[h1,yb*tf.ones([self.batch_size,7,7,self.y_dim])])  # [64,7,7,64*2+10]

            h2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(h1,64*2,5,(2,2),'SAME',
                                                                                     kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                                                                     name='g_h2_convtrans'),training=training,name='g_h2_bn'),name='g_h2_relu')  # [64,14,14,64*2]
            h2 = tf.concat(axis=3, values=[h2,yb*tf.ones([self.batch_size,14,14,self.y_dim])]) # [64,14,14,64*2+10]

            return tf.nn.sigmoid(
                tf.layers.conv2d_transpose(h2, 1, 5,(2,2),'SAME',
                                           kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                           name='g_h3_convtrans'))  # n, 28, 28, 1


    def generator_v3(self,z,y,training=False):

        with tf.variable_scope("generator") as scope:
            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])  # n,1,1,10
            z = tf.concat([z, y], 1)  # 64, 110

            h0 = tf.nn.relu(
                self.g_bn0(linear(z, 1024, 'g_h0_lin',stddev=self.stddev),training=training),name='g_h0_relu')  # n,1024
            h0 = tf.concat([h0, y], 1)  # [64,1034]

            h1 = tf.nn.relu(self.g_bn1(
                linear(h0, 64 * 2 * 7 * 7, 'g_h1_lin',stddev=self.stddev),training=training),name='g_h1_relu')  # [64,64*2*7*7]
            h1 = tf.reshape(h1, [self.batch_size, 7, 7, 64 * 2])  # [64,7,7,64*2]
            h1 = tf.concat(axis=3, values=[h1,yb*tf.ones([self.batch_size,7,7,self.y_dim])])  # [64,7,7,64*2+10]

            h2 = tf.nn.relu(self.g_bn2(tf.layers.conv2d_transpose(h1,64*2,5,(2,2),'SAME',
                                                                  kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                                                  name='g_h2_convtrans'),
                                       training=training),name='g_h2_relu')  # [64,14,14,64*2]
            h2 = tf.concat(axis=3, values=[h2,yb*tf.ones([self.batch_size,14,14,self.y_dim])]) # [64,14,14,64*2+10]

            return tf.nn.sigmoid(
                tf.layers.conv2d_transpose(h2, 1, 5,(2,2),'SAME',
                                           kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                           name='g_h3_convtrans'))  # n, 28, 28, 1


    def generator_v4(self,z,y,training=False):
        # z [N,100]
        # y [N,10]
        with tf.variable_scope('generator') as scope:
            # label扩张维度，输入串联条件
            yb = tf.reshape(y,[self.batch_size,1,1,self.y_dim])  # [N,10]->[N,1,1,10]
            z = tf.concat(values=[z,y],axis=1)   # [N,110]

            # 全链接，BN，激活，串联条件
            h0 = tf.nn.relu(self.g_bn0(linear(z,self.filters[0],'g_h0_lin'),
                                       training=training),
                           name='g_h0_relu')  #[N,1024]
            h0 = tf.concat(values=[h0,y],axis=1)  # [N,1024+10]

            # 全链接，BN，激活，reshape，串联条件
            h1 = tf.nn.relu(self.g_bn1(linear(h0,self.filters[1]*7*7,'g_h1_lin'),
                                       training=training),
                            name='g_h1_relu')  #[N,128*7*7]
            h1 = tf.reshape(h1,[self.batch_size,7,7,self.filters[1]])  #[N,7,7,128]
            h1 = tf.concat(axis=3,values=[h1,yb*tf.ones([self.batch_size,7,7,self.y_dim])])  #[N,7,7,128+10]

            # 转置卷积，BN，激活，串联条件
            h2 = tf.layers.conv2d_transpose(h1,self.filters[2],5,(2,2),'SAME',
                                            kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),name='g_h2_convtrans')
            h2 = tf.nn.relu(self.g_bn2(h2,training=training),name='g_h2_relu')  #[N,14,14,64]
            h2 = tf.concat(values=[h2,yb*tf.ones([self.batch_size,14,14,self.y_dim])],axis=3)  #[n,14,14,64+10]

            # 转置卷积
            h3 = tf.layers.conv2d_transpose(h2,self.c_dim,5,(2,2),'SAME',kernel_initializer=tf.random_normal_initializer(stddev=self.stddev),name='g_h3_convtrans')
            # 归一道0,1之间
            h4 = tf.nn.sigmoid(h3,name='g_h4_sig')
            return h4

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.img_size, self.img_size,self.version)

    def save(self, checkpoint_dir, step):
        model_name = "CDCGAN_MNIST.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        if os.path.isdir('/home/xuhaiyue'):
            # 目前在服务器上
            checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        elif os.path.isdir('results/'+checkpoint_dir):
            # 本机运行
            checkpoint_dir = os.path.join('results/'+checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def load_mnist(self):
        # data_dir = os.path.join("./data", 'MNIST_data')  # 注意这里的路径！！！！！
        dir = "MNIST_data"
        if not os.path.isdir(dir):
            dir = "/home/xuhaiyue/ssd/data/MNIST_data"
        data_dir = os.path.join(dir)
        fd = open(os.path.join(data_dir, 'train-images.idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int)

        seed = 1024
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        return X / 255., y_vec