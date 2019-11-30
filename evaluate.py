import tensorflow as tf
import numpy as np
import config.config as cfg
import datetime
import SimpleITK as sitk
import os
import core.model_factory as model_factory
from tools import utils
from core.loss_func import softmax_loss_function
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

#name = 'resnet_aspp_5fold'
#name = 'xception_aspp'
#name = 'cnn_v2_softmax'
name = 'vnet_dice'
#cfg_name = 'resnet_aspp_5fold'
#scores = ['0.005089','0.005848','0.005822','0.004236','0.005070']
scores = ['0.001648']
fold_num = 3
threshold = 0.5
data_path = cfg.data_path
output_path = './result/%s/'%name
predict_tail = cfg.predict_tail
#checkpoint_dir = '/home/data_new/guofeng/projects/Segmentation/NPCCode/result/resnet_aspp_5fold/model/'
predict_op = 'softmax'
def evaluate(fold, score):
    """evaluate model by stepwise moving along the 3D image"""
    patient_list = np.load('./result/%s/data/data.npy'%(name))
    val_size = 24
    pat_iter = (fold - 1) * val_size
    val_set = patient_list[pat_iter:pat_iter + val_size]

    tf.reset_default_graph()
    model = model_factory.modelMap[name]
    print(name)
    provider = utils.DataProvider()
    is_training = tf.Variable(tf.constant(False))
    provider.test_list = val_set
    #print(provider.test_list)
    test_holder = provider.get_test_holder()
    model_instance = model(is_training)

    predict_label = model_instance.inference_op(test_holder)

    model_name = 'model_%d_%s' % (fold, score)
    model_path = './result/%s/model/%s' % (name, model_name)
    model_final_path = './result/%s/model/model_saved_%d' % (name, fold)
    with tf.Session() as sess:
        print("{}: Start evaluation...".format(datetime.datetime.now()))
        saver = tf.train.Saver()
        saver.restore(sess, model_final_path)
        #imported_meta.restore(sess, tf.train.latest_checkpoint(checkpoint_dir,latest_filename="model_1_0.1111"))
        print("{}: Restore checkpoint success".format(datetime.datetime.now()))

        output_dir = os.path.join(output_path, "predict")
        pro_dir0 = os.path.join(output_path, "probability0")
        pro_dir1 = os.path.join(output_path, "probability1")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(pro_dir0):
            os.makedirs(pro_dir0)
        if not os.path.exists(pro_dir1):
            os.makedirs(pro_dir1)
        provider.init_test()
        while True:
            test_lst = provider.get_test_value()
            if test_lst is None:
                break
            output_lst = []
            for list in test_lst:
                #prob = sess.run(predict_label[0], feed_dict={test_holder: list})
                output = sess.run(predict_label[1], feed_dict={test_holder: list})
                softmax_prob = tf.nn.softmax(logits=predict_label[0], name='softmax_prob')
                softmax_output = sess.run(softmax_prob, feed_dict={test_holder: list})
                output_lst.append(output)
            patient_name = provider.test_list[provider.test_iter]
            print(patient_name)
            patient_img = sitk.ReadImage(os.path.join(data_path, patient_name + cfg.patient_file_tail))

            if predict_op == 'softmax':
                predicted_label = output_lst[0][:, :103, :198, :]
                predicted_label = np.squeeze(predicted_label)
                predicted_label = np.array(predicted_label, dtype=np.uint8)
            else:
                predicted_label = output_lst[0][:, :103, :198, :]
                predicted_label = np.squeeze(predicted_label)
                predicted_label = np.array(predicted_label + threshold, dtype=np.uint8)

            '''count = 0
            for i in range(103):
                for j in range(198):
                    for k in range(160):
                        if predicted_label[i][j][k] > 0:
                            count += 1'''
            '''pat = patient_name
            seg_path = os.path.join(data_path, pat + cfg.patient_seg_tail)
            seg_arr = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
            #print(np.shape(seg_arr),np.shape(output_lst[0][:, :103, :198, :]))
            correct_pred = tf.equal(output_lst[0][:, :103, :198, :], tf.cast(np.squeeze(seg_arr), dtype=tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            num = tf.reduce_sum(output_lst[0][:, :103, :198, :])/tf.reduce_sum(tf.cast(seg_arr,dtype=tf.int64))
            accuracy,num = sess.run([accuracy,num])
            print(accuracy,num)'''

            '''from tensorflow.python.framework import ops
            seg_arr = tf.expand_dims(tf.cast(ops.convert_to_tensor(seg_arr),dtype=tf.int64),axis=0)
            print(predict_label[0],seg_arr)
            #segarr = sess.run(seg_arr)

            prob = ops.convert_to_tensor(prob)
            print(prob,seg_arr)
            loss = softmax_loss_function(prob,seg_arr,cfg.use_dst_weight)

            loss = sess.run(loss)
            print(loss)'''

            outprob = sitk.GetImageFromArray(np.squeeze(softmax_output[:, :103, :198, :, 0]))
            outprob.SetSpacing(patient_img.GetSpacing())
            outprob.SetDirection(patient_img.GetDirection())
            outprob.SetOrigin(patient_img.GetOrigin())
            sitk.WriteImage(outprob, os.path.join(pro_dir0, patient_name + predict_tail))
            outprob = sitk.GetImageFromArray(np.squeeze(softmax_output[:, :103, :198, :, 1]))
            outprob.SetSpacing(patient_img.GetSpacing())
            outprob.SetDirection(patient_img.GetDirection())
            outprob.SetOrigin(patient_img.GetOrigin())
            sitk.WriteImage(outprob, os.path.join(pro_dir1, patient_name + predict_tail))

            outseg = sitk.GetImageFromArray(predicted_label)
            outseg.SetOrigin(patient_img.GetOrigin())
            outseg.SetDirection(patient_img.GetDirection())
            outseg.SetSpacing(patient_img.GetSpacing())
            sitk.WriteImage(outseg, os.path.join(output_dir, patient_name + predict_tail))
            provider.test_iter += 1
for i in range(1, fold_num + 1):
    #evaluate(fold=i,score=scores[i-1])
    evaluate(fold=i,score=0)