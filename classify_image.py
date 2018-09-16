from __future__ import absolute_import
from __future__ import division
from __future__ import  print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import  urllib
import tensorflow as tf

tf.app.flags.DEFINE_string(
    'image_file', '',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_string(
    'model_file', '',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_string(
    'label_file', '',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer('num_top_predicitons', 5, '')

FLAGS = tf.app.flags.FLAGS

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


class NodeLookup(object):

    def __init__(self, label_path=None):
        if not label_path:
            tf.logging.fatal('please specify the label file.')
            return
        self.node_lookup = self.load(label_path)

    def load(self, label_path):
        if not tf.gfile.Exists(label_path):
            tf.logging.fatal('File dose not exist %s', label_lookup_path)

        proto_as_ascii_lines = tf.gfile.GFile(label_path).readlines()
        id_to_human = {}
        for line in proto_as_ascii_lines:
            if line.find(':') < 0:
                continue
            _id, human = line.rstrip('\n').split(':')
            id_to_human[int(_id)] = human

        return id_to_human

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return  self.node_lookup[node_id]


def create_graph(model_file=None):
    if not model_file:
        model_file = FLAGS.model_file
    with open(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inferece_on_image(image, model_file=None):
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exits %s', image)

    image_data = open(image, 'rb').read()

    create_graph(model_file)

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('CifarNet/Predictions/Softmax:0')
        predictions = sess.run(softmax_tensor, {'input:0': image_data})
        predictions = np.squeeze(predictions)

        node_lookup = NodeLookup(FLAGS.label_file)

        top_k = predictions.argsort()[-FLAGS.num_top_predicitons:][::-1]
        top_names = []
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            top_names.append(human_string)
            score = predictions[node_id]
            print('id:[%d] name:[%s] (score = %.5f)' % (node_id, human_string, score))
        return  predictions, top_k, top_names


if __name__ == '__main__':
    run_inferece_on_image(image=FLAGS.image_file, model_file=FLAGS.model_file)
