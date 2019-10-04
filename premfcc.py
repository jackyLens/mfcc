import tensorflow as tf
# print(tf.__version__)
# import models
import numpy as np
import os
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# import matplotlib.pyplot as plt
# from tensorflow.python.framework import graph_util

path1 = '/Users/zhenghuimin/1/'


# path1 = 'F:/fenge/test3/'
# pb_path = 'E:/test/mfcc.pb'


def load_wav_file(filename):
    """Loads an audio file and returns a float PCM-encoded array of samples.
    Args:
      filename: Path to the .wav file to load.
    Returns:
      Numpy array holding the sample data as floats between -1.0 and 1.0.
    """
    with tf.Session(graph=tf.Graph()) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)

        return sess.run(wav_decoder, feed_dict={wav_filename_placeholder: filename}).audio.flatten()


def compute_mfcc(wavdata):
    with tf.Session(graph=tf.Graph()) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        wav_data_placeholder = tf.placeholder(tf.float32, [None, None], name='input')
        wav_clip_data_placeholder = tf.clip_by_value(wav_data_placeholder, -1.0, 1.0)
        spectrogram = contrib_audio.audio_spectrogram(
            wav_clip_data_placeholder,
            window_size=400,
            stride=160,
            magnitude_squared=True)
        mfcc_ = contrib_audio.mfcc(spectrogram, 16000, upper_frequency_limit=7500, lower_frequency_limit=75,
                                   dct_coefficient_count=64, filterbank_channel_count=64, name='mfccs')
        # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["input", 'mfccs'])
        # with tf.gfile.FastGFile(pb_path, mode='wb') as f:
        #     f.write(constant_graph.SerializeToString())
        return sess.run(mfcc_, feed_dict={wav_data_placeholder: wavdata})


def get_wav_files(wav_path):
    wav_files = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith(".wav") or filename.endswith(".WAV"):
                filename_path = os.path.join(dirpath, filename)
                # if os.stat(filename_path).st_size < 240000:
                #     continue
                wav_files.append(filename_path)

    return wav_files


# wave0 = get_wav_files(path0)
wave1 = get_wav_files(path1)
file = open(path1 + 'mfcc.txt', 'w')
for i in range(1, len(wave1) + 1):
    # s = load_wav_file(wave1[i])

    s = load_wav_file(path1 + str(i) + '.wav')
    tmp = np.array(s[0:16000])
    tmp = tmp.reshape((len(tmp)), 1)
    mfcc = compute_mfcc(tmp)

    tmparray = np.array(mfcc).reshape(1, mfcc.shape[1] * mfcc.shape[-1]).tolist()
    print(i, path1 + str(i) + '.wav')
    file.write(str(tmparray))
    file.write('\n')
file.close()
