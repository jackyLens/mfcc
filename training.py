import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import numpy as np
import readmfcc
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import graph_util
# import mobilenetv2


# pbpath = 'E:/0/dscnn.pb'
# read data
path0 = '/Users/zhenghuimin/Downloads/mfcc_snore.txt'


path1 = '/Users/zhenghuimin/Downloads/mfcc_nonsnore.txt'

hud = []
cui = []
yh, ud = readmfcc.readdata(hud, cui, path0, 0)


x, y = readmfcc.readdata(yh, ud, path1, 1)

train_data, test_data, train_lables, test_lables = train_test_split(x, y, test_size=0.05, random_state=2600)
NUM = len(train_lables)
print(NUM)
print(len(test_lables))


def get_test_data(testdata):
    test_mfcc = []
    for i in range(len(testdata)):
        dsd = np.array(testdata[i])
        dsd = dsd.reshape((1, len(dsd)))
        test_mfcc.append(dsd)
    return test_mfcc


def get_batches(iterations, x, y, batch_size):
    length = len(x)
    batches_wavs = []
    batches_labels = []
    if iterations * batch_size < length:
        for j in range(batch_size):
            dsd = x[(iterations - 1) * batch_size + j]
            batches_wavs.append(dsd)
            batches_labels.append(y[(iterations - 1) * batch_size + j])
    else:
        for j in range(batch_size):
            dsd = x[(iterations - 1) * batch_size + j - length]
            batches_wavs.append(dsd)
            batches_labels.append(y[(iterations - 1) * batch_size + j - length])
    return batches_wavs, batches_labels


def one_hot(labels):
    sess = tf.Session()
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, labels], 1)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, 2]), 1, 0)
    xx = sess.run(onehot_labels)
    return xx


def ds_cnnmodel(fingerprint_input, model_size_info, is_training):
    def ds_cnn_arg_scope(weight_decay=0):
        """Defines the default ds_cnn argument scope.
        Args:
          weight_decay: The weight decay to use for regularizing the model.
        Returns:
          An `arg_scope` to use for the DS-CNN model.
        """
        with slim.arg_scope(
                [slim.convolution2d, slim.separable_convolution2d],
                weights_initializer=slim.initializers.xavier_initializer(),
                biases_initializer=slim.init_ops.zeros_initializer(),
                weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
            return sc

    def _depthwise_separable_conv(inputs,
                                  num_pwc_filters,
                                  sc,
                                  kernel_size,
                                  stride):
        """ Helper function to build the depth-wise separable convolution layer.
        """

        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                      num_outputs=None,
                                                      stride=stride,
                                                      depth_multiplier=1,
                                                      kernel_size=kernel_size,
                                                      scope=sc + '/depthwise_conv')

        bn = slim.batch_norm(depthwise_conv, scope=sc + '/dw_batch_norm')
        pointwise_conv = slim.convolution2d(bn,
                                            num_pwc_filters,
                                            kernel_size=[1, 1],
                                            scope=sc + '/pointwise_conv')
        bn = slim.batch_norm(pointwise_conv, scope=sc + '/pw_batch_norm')
        return bn

    label_count = 2
    input_frequency_size = 64
    input_time_size = 98
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])

    t_dim = input_time_size
    f_dim = input_frequency_size

    # Extract model dimensions from model_size_info
    num_layers = model_size_info[0]
    conv_feat = [None] * num_layers
    conv_kt = [None] * num_layers
    conv_kf = [None] * num_layers
    conv_st = [None] * num_layers
    conv_sf = [None] * num_layers
    i = 1
    for layer_no in range(0, num_layers):
        conv_feat[layer_no] = model_size_info[i]
        i += 1
        conv_kt[layer_no] = model_size_info[i]
        i += 1
        conv_kf[layer_no] = model_size_info[i]
        i += 1
        conv_st[layer_no] = model_size_info[i]
        i += 1
        conv_sf[layer_no] = model_size_info[i]
        i += 1

    scope = 'DS-CNN'
    with tf.variable_scope(scope) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                            activation_fn=None,
                            weights_initializer=slim.initializers.xavier_initializer(),
                            biases_initializer=slim.init_ops.zeros_initializer(),
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.batch_norm],
                                is_training=is_training,
                                decay=0.96,
                                updates_collections=None,
                                activation_fn=tf.nn.relu):
                for layer_no in range(0, num_layers):
                    if layer_no == 0:
                        net = slim.convolution2d(fingerprint_4d, conv_feat[layer_no], \
                                                 [conv_kt[layer_no], conv_kf[layer_no]],
                                                 stride=[conv_st[layer_no], conv_sf[layer_no]], padding='SAME',
                                                 scope='conv_1')
                        net = slim.batch_norm(net, scope='conv_1/batch_norm')
                    else:
                        net = _depthwise_separable_conv(net, conv_feat[layer_no], \
                                                        kernel_size=[conv_kt[layer_no], conv_kf[layer_no]], \
                                                        stride=[conv_st[layer_no], conv_sf[layer_no]],
                                                        sc='conv_ds_' + str(layer_no))
                    t_dim = math.ceil(t_dim / float(conv_st[layer_no]))
                    f_dim = math.ceil(f_dim / float(conv_sf[layer_no]))

                net = slim.avg_pool2d(net, [t_dim, f_dim], scope='avg_pool')

        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
        logits = slim.fully_connected(net, label_count, activation_fn=None, scope='fc1')
        return logits


check_nans = False
epochs = 40
batch_size = 64
fingerprint_input = tf.placeholder(tf.float32, [None, 98, 64], name='fingerprint_input')
ground_truth_input = tf.placeholder(tf.float32, [None, 2], name='groundtruth_input')
# istraining = tf.placeholder(tf.bool, None, name='state')
# istraining = tf.constant(False, isdtype=bool, shape=[], name='state')
istraining = tf.Variable(False, name='training', trainable=False)

logits = ds_cnnmodel(fingerprint_input, [5, 64, 10, 4, 2, 2, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1], istraining)
# logits = ds_cnnmodel(fingerprint_input, [5, 172, 10, 4, 2, 1, 172, 3, 3, 2, 2, 172, 3, 3, 1, 1, 172, 3, 3, 1, 1,172, 3, 3, 1, 1], istraining)
# logits = mobilenetv2.mobilev2(fingerprint_input, 1, traing=istraining)
control_dependencies = []
if check_nans:
    checks = tf.add_check_numerics_ops()
    control_dependencies = [checks]

with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_input, logits=logits))
tf.summary.scalar('cross_entropy', cross_entropy_mean)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.name_scope('train'), tf.control_dependencies(update_ops), tf.control_dependencies(control_dependencies):
    learning_rate_input = tf.placeholder(tf.float32, [], name='learning_rate_input')
    train_op = tf.train.AdamOptimizer(learning_rate_input)
    train_step = slim.learning.create_train_op(cross_entropy_mean, train_op)

predicted_indices = tf.argmax(logits, 1, name='prediction')
expected_indices = tf.argmax(ground_truth_input, 1)
correct_prediction = tf.equal(predicted_indices, expected_indices)
confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices, num_classes=2)
evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', evaluation_step)

global_step = tf.train.get_or_create_global_step()
increment_global_step = tf.assign(global_step, global_step + 1)

saver = tf.train.Saver(tf.global_variables())
merged_summaries = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('C:/Users/zhenghuimin/train', sess.graph)
    # validation_writer = tf.summary.FileWriter('/validation')
    params = tf.trainable_variables()
    num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
    print('Total number of Parameters: ', num_params)
    best_accuracy = 0
    training_steps_max = 3000
    for e in range(epochs):
        train_data1, test_data1, train_lables1, test_lables1 = train_test_split(train_data, train_lables, test_size=0,
                                                                                random_state=e)
        for iteration in range(int(NUM / batch_size) + 1):
            inputx, inputy = get_batches(iteration, train_data1, train_lables1, batch_size)
            inputx = np.array(inputx).reshape((-1, 98, 64))
            inputy1 = one_hot(inputy)
            feed = {fingerprint_input: inputx, ground_truth_input: inputy1, learning_rate_input: 0.0002,
                    istraining: True}
            train_summary, loss_mean, _, accuracy = sess.run(
                [merged_summaries, cross_entropy_mean, train_step, evaluation_step], feed_dict=feed)
            train_writer.add_summary(train_summary, e * int(NUM/batch_size) + iteration)
            if (iteration + 1) % 2 == 0:
                print("Epoch: {}/{}".format(e, epochs), "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss_mean), "Train acc: {:.6f}".format(accuracy))
                # Compute validation loss at every 10 iterations
                if (iteration + 1) % 20 == 0:
                    valy = one_hot(test_lables)
                    print(test_lables)
                    ud = get_test_data(test_data)
                    ux = np.array(ud).reshape((-1, 98, 64))
                    feed = {fingerprint_input: ux, ground_truth_input: valy, istraining: False}
                    loss_v, acc_v, pl, matrix = sess.run(
                        [cross_entropy_mean, evaluation_step, predicted_indices, confusion_matrix], feed_dict=feed)
                    print("Epoch: {}/{}".format(e, epochs), "Iteration: {:d}".format(iteration),
                          "Validation loss: {:6f}".format(loss_v), "Validation acc: {:.6f}".format(acc_v))
                    print('confusion matric:')
                    print(matrix)
                    if acc_v >= best_accuracy:
                        gd = sess.graph.as_graph_def()
                        for node in gd.node:
                            if node.op == 'RefSwitch':
                                node.op = 'Switch'
                                for index in range(len(node.input)):
                                    if 'moving_' in node.input[index]:
                                        node.input[index] = node.input[index] + '/read'
                            elif node.op == 'AssignSub':
                                node.op = 'Sub'
                                if 'use_locking' in node.attr:
                                    del node.attr['use_locking']
                        constant_graph = graph_util.convert_variables_to_constants(sess, gd,
                                                                                   ['fingerprint_input', 'prediction'])
                        tf.train.write_graph(constant_graph, '/Users/zhenghuimin/', 'modeltest1.pb', as_text=False)
