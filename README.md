# dcgan-cdcgan[lsun-celebA-MNIST]

paper:[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

code reference：[here](https://github.com/carpedm20/DCGAN-tensorflow)


# Prerequisites
python2.7

tensorflow1.0.1

nunmpy1.12.1

scipy0.19.0

pillow4.1.0

# Dataset

LSUN:conference_room

[celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

MNIST

# Data preprocessing

Run an face detector in [Openface](https://cmusatyalab.github.io/openface/) on these images in celebA

with docker, you can enter the following code:
```python
# Ubuntu16.04
$ docker pull bamos/openface  # only once

# mount local folder to docker, every time
$ docker run -p 9000:9000 -p 8000:8000 -it -v /local/absolute/path:/docker/path bamos/openface bin/bash
# or dont mount local folder to docker,every time
$ docker run -p 9000:9000 -p 8000:8000 -t -i bamos/openface bin/bash

# detect face
$ cd /root/openface
$ ./util/align-dlib.py /docker/path align innerEyesAndBottomLip / aligned_face --size 64
# You can also replace outerEyesAndNose with innerEyesAndBottomLip. Two alignment methods

```

# Results

## celebA

0~23 epochs
![after 23 epoch](https://github.com/xhygh/dcgan-cdcgan-lsun-celebA-MNIST-/blob/master/img/celebA_23epoch.gif?raw=true)


## MNIST

I use tf.layers.batch_normalization and tf.contib.layers.batch_norm and 2 different network architectures.

mnist_v1 & mnist_v4 use a network architecture and mnist_v2 & mnist_v3 use another one.

0~20epochs of mnist_v1
![mnist_v1](https://github.com/xhygh/dcgan-cdcgan-lsun-celebA-MNIST-/blob/master/img/mnist_v1_epoch20.gif?raw=true)

0~20epochs of mnist_v2
![mnist_v2](https://github.com/xhygh/dcgan-cdcgan-lsun-celebA-MNIST-/blob/master/img/mnist_v2_epoch20.gif?raw=true)

0~20epochs of mnist_v3
![mnist_v3](https://github.com/xhygh/dcgan-cdcgan-lsun-celebA-MNIST-/blob/master/img/mnist_v3_epoch20.gif?raw=true)

0~20epochs of mnist_v4
![mnist_v4](https://github.com/xhygh/dcgan-cdcgan-lsun-celebA-MNIST-/blob/master/img/mnist_v4_epoch20.gif?raw=true)


#There are something wrong with tf.layers.batch_nomalization.#




