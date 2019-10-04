import tensorlayer as tl
import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base,inception_v3,inception_v3_arg_scope
import skimage
import skimage.io
import skimage.transform
import time,os
from data.imagenet_classes import *


def load_image(path):
    image = skimage.io.imread(path)
    img = image/255.0
    assert (0<=img).all() and (img<=1.0).all()
    short_edge = min(img.shape[2])
    yy = int((img.shape[0]-short_edge)/2)
    xx = int((img.shape[1]-short_edge)/2)
    corp_img = img[yy:yy+short_edge,xx:xx+short_edge]
    resized_img = skimage.transform.resize(corp_img,(299,299))
    return resized_img


x = tf.placeholder(tf.float32,shape=[None,299,299,3])
net_in = tl.layers.InputLayer(x,name='input_layer')
with slim.arg_scope(inception_v3_arg_scope()):
    network = tl.layers.SlimNetsLayer(layer=net_in,slim_layer=inception_v3,
                                      slim_args={'num_classes':10001,
                                                 'is_training':False},name='Inception3')


