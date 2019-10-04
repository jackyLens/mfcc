import tensorflow as tf
import numpy as np
import readmfcc


# pb_file_path = '/Users/zhenghuimin/modeltest1.pb'
pb_file_path = 'model05.pb'
# pathtest = 'E:/val/ceshi/20181101142907/mfcc.txt'      #model9 略优于 model10
# pathtest = 'E:/val/chanxianWoman/20181029001004/mfcc.txt'  #model9 = model10
# pathtest = 'E:/val/chanxianWomanX/20181028235146/mfcc.txt'   #model9  略优于  model10
# pathtest = 'E:/val/fangzs/mfcc.txt'                           #model9  约等于  model10
# pathtest = 'E:/val/hebo/20181028022406/mfcc.txt'                #model9 略低于 model10
# pathtest = 'E:/val/i51030/20181029232811/mfcc.txt'                 #model9 = model10
# pathtest = 'E:/val/kupeng/20181028224746/mfcc.txt'                   #model9 优于 model10
# pathtest = 'E:/val/liuming/20181028012557/mfcc.txt'                   #model9 < model10
# pathtest = 'E:/val/liuming/20181027020659/mfcc.txt'                    #model9 略等于 model10
# pathtest = 'E:/val/maosq/20181025235758/mfcc.txt'                         #model9 略等于 model10
# pathtest = 'E:/val/meih/mfcc.txt'                                              #model9 略等于 model10
# pathtest = 'E:/val/menglg/20181026000838/mfcc.txt'                                 #model9 略等于 model10
# pathtest = 'E:/val/se1030/20181029232308/mfcc.txt'                                    #model9 略等于 model10
# pathtest = 'E:/val/wangj/mfcc.txt'                                                       #model9 等于 model10
# pathtest = 'E:/val/woman/cop/mfcc.txt'                                                      #model9 略等于 model10
# pathtest = 'E:/val/zhousx/20181107000448/mfcc.txt'
# pathtest = 'E:/val/zhousx/20181107005035/mfcc.txt'
# pathtest = 'E:/val/zout/20181026000000/mfcc.txt'
# pathtest = 'E:/val/zour/mfcc.txt'
# pathtest = 'E:/val/xiaoq/20181029235108/mfcc.txt'
pathtest = '/Users/zhenghuimin/Downloads/mfcc2.txt'


def recognize(pb_file_path,ss):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            input_x = sess.graph.get_tensor_by_name("fingerprint_input:0")
            state = sess.graph.get_tensor_by_name("state:0")
            out_label = sess.graph.get_tensor_by_name("prediction:0")
            ux = np.array(ss).reshape((-1, 98, 64))
            feed = {input_x: ux,state:False}
            label = sess.run(out_label, feed_dict=feed)
            np.savetxt('/Users/zhenghuimin/1/val3.txt',label)
            print('predicted label is :', label)


def get_test_data(mfccdata):
    test_mfcc = []
    for i in range(len(mfccdata)):
        dsd = np.array(mfccdata[i])
        dsd = dsd.reshape((1, len(dsd)))
        test_mfcc.append(dsd)
    return test_mfcc

hud = []
cui = []
yh,ud = readmfcc.readdata(hud, cui, pathtest, 0)

datain = get_test_data(yh)
recognize(pb_file_path,datain)



