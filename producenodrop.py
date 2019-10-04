from __future__ import print_function
import tensorflow as tf
from tensorflow.core.framework import graph_pb2


def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('%d %s %s' % (i, node.name, node.op))
        [print(u'└─── %d ─ %s' % (i, n)) for i, n in enumerate(node.input)]


graph = tf.GraphDef()
with tf.gfile.Open('/Users/zhenghuimin/modeltest1.pb', 'rb') as f:
    data = f.read()
graph.ParseFromString(data)
display_nodes(graph.node)