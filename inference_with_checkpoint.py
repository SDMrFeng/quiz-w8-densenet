from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np

tf.app.flags.DEFINE_string('dataset_name', 'cifar10', '')
tf.app.flags.DEFINE_string('dataset_dir', '/home/fengxuezhi/tmp/cifar10', '')

tf.app.flags.DEFINE_string('model_name', 'cifarnet', '')
tf.app.flags.DEFINE_string('output_file', './output.pb', '')

tf.app.flags.DEFINE_string('checkpoint_path', '/home/fengxuezhi/tmp/cifarnet-model/', '')

tf.app.flags.DEFINE_string('pic_path', './test.jpg', '')

FLAGS = tf.app.flags.FLAGS

is_training = False
preprocessing_name = FLAGS.model_name

graph = tf.Graph().as_default()

dataset = dataset_factory.get_dataset(FLAGS.dataset_name, 'train',
                                      FLAGS.dataset_dir)

image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    preprocessing_name,
    is_training = False)

network_fn = nets_factory.get_network_fn(
    FLAGS.model_name,
    num_classes=dataset.num_classes,
    is_training=is_training)

if hasattr(network_fn, 'default_image_size'):
    image_size = network_fn.default_image_size
else:
    image_size = FLAGS.default_image_size

placeholder = tf.placeholder(name='input', dtype=tf.string)
image = tf.image.decode_jpeg(placeholder, channels=3)
iamge = image_preprocessing_fn(image, image_size, image_size)
image = tf.expand_dims(iamge, 0)
logit, endpoints = network_fn(image)

saver = tf.train.Saver()
sess = tf.Session()
checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
saver.restore(sess, checkpoint_path)
image_value = open(FLAGS.pic_path, 'rb').read()
logit_value = sess.run([logit], feed_dict={placeholder:image_value})
print(logit_value)
print(np.argmax(logit_value))

