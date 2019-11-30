import os
import SimpleITK as sitk
import config.config as cfg
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
'''data_path = cfg.data_path
patient_tail = cfg.predict_tail

pat = './result/cnn_v2/predict/HAN_XIU_LING_0013550587_.mha'

train_path = os.path.join(pat)
pat_arr = sitk.GetArrayFromImage(sitk.ReadImage(train_path))
print(pat_arr.shape)
count = 0
for i in range(103):
    for j in range(198):
        for k in range(160):
            if pat_arr[i][j][k] > 0:
                count += 1'''
#print(count)
#pat_arr = pat_arr[:,:,:,np.newaxis]
#print(pat_arr)
model = 'vnet'
'''data_list = []
for i in range(1,5+1):
    data = np.load('/home/data_new/guofeng/projects/Segmentation/NPCCode/result/%s/data/data_%d.npy'%(model,i))
    print(data)
    for j in range(len(data)):
        data_list.append(data[j])'''
    #print(data_list)
#np.save('./result/%s/data/data.npy' % (model), data_list)
#data1 = np.load('./result/%s/data/data_1.npy' % (model))
#print(len(data1),data1)
def crop_tensor(input_tensor):

    data_shape = input_tensor.get_shape()
    #transed_shape = [int(data_shape[0]), int(data_shape[1]), int(data_shape[2]), int(data_shape[3])]
    central_x = int(data_shape[1]) // 2
    central_y = int(data_shape[2]) // 2
    central_z = int(data_shape[3]) // 2

    labels_slice = tf.slice(input_tensor, [0, central_x, central_y, central_z],
                            [1, int(data_shape[1]) - central_x, int(data_shape[2]) - central_y,
                             int(data_shape[3]) - central_z])
    print(labels_slice)
'''pat = './result/comp_result/deeplab_5fold_weights/predict/HAN_XIU_LING_0013550587_.mha'

train_path = os.path.join(pat)
pat_arr = sitk.GetArrayFromImage(sitk.ReadImage(train_path))
pat_arr = ops.convert_to_tensor(pat_arr)
pat_arr = tf.expand_dims(pat_arr,0)
print(pat_arr.shape)'''
#crop_tensor(pat_arr)

'''data = np.load('./result/vnet_dice/data/data.npy')
#patient_list2 = np.load('./result/vnet_eight/data/data.npy')
print(data)
patient_list = []
for i in range(len(data)):
    patient_list.append(data[i])
val_size = len(patient_list) // 5
fold_iter = 4
pat_iter = (fold_iter - 1) * val_size
val_set = patient_list[pat_iter:pat_iter + val_size]
print(val_set)'''
total_pw = []
'''for i in range(len(patient_list)):
    seg_path = os.path.join(cfg.data_path, patient_list[i] + cfg.patient_seg_tail)
    seg_arr = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
    print(np.mean(seg_arr))
    pw = (1 - np.mean(seg_arr)) / np.mean(seg_arr)
    print(i,pw)
    total_pw.append(pw)'''
#np.save('pw.npy',total_pw)
def pw():
    pw = np.load('pw.npy')
    #new = pw / (np.sum(pw)/len(pw))
    #new1 = pw / len(pw)
    #new2 = np.power(pw,1/2)
    print(pw)
    new2 = pw / 4
    new4 = pw/8
    new8 = pw / 12
    snew = np.sqrt(pw)
    print(new2)
    print(new4)
    print(new8)
    print(snew)
    #data = np.load('./result/vnet_sqrt/data/data.npy')
    #print(data[23])
def move_score():
    import shutil
    data_path = "./result/comp_result/xception_aspp_softmax_jacc/data/data.npy"
    score_path = './result/comp_result/xception_aspp_softmax_jacc/score'
    data = np.load(data_path)
    print(data)
    print('load data successfully!!!!!!!!!!!')
    patient_list = []
    for i in range(len(data)):
        patient_list.append(data[i])
    val_size = len(patient_list) // 5
    # fold_iter = cfg.fold_iter
    for i in range(1,5+1):
        dst_path = './result/comp_result/xception_aspp_softmax_jacc/score%d'%i
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        pat_iter = (i-1) * val_size
        val_set = patient_list[pat_iter:pat_iter + val_size]
        for j in range(len(val_set)):
            ori_json = os.path.join(score_path,val_set[j] + '.json')
            print(ori_json)
            shutil.copy(ori_json,dst_path)
#move_score()
#import random
#if random.randint(0,99) < 50:
#    print(1)
def comput_two_seg():
    # caculate score of dice
    truth_path = '/home/gf/guofeng/all_120/two_seg/Philips_1st_seg_processed/'
    output_path = '/home/data_new/guofeng/projects/Segmentation/NPCCode/two_seg/score/'
    predict_path = '/home/gf/guofeng/all_120/two_seg/Philips_2nd_seg_processed/'
    score_bin_dir = r"/home/gf/guofeng/ImageValidationISLES/bin"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    #tpo = scorer.create_t_p_o_list(truth_path, predict_path, output_path)
    t_p_o_list = []
    predict_list = os.listdir(predict_path)
    #print(predict_list)
    truth_list = os.listdir(truth_path)
    truth_seg_tail = "MRSegRAIResampleROIReg.mha"
    pred_seg_tail = 'MRAnotherSegRAIResampleROIReg.mha'
    for pred in predict_list:
        for truth in truth_list:
            if pred == truth:
                t = truth_path + '%s/MR/'%truth + truth + truth_seg_tail
                p = predict_path + '%s/MR/'%pred + pred + pred_seg_tail
                o = output_path + truth + ".json"
                t_p_o_list.append((t, p, o))
    print(t_p_o_list)
    os.chdir(score_bin_dir)
    for i in t_p_o_list:
        truth = i[0]
        predict = i[1]
        output = i[2]
        os.system("./ImageValidationISLES " + truth + " " + predict + " " + output)
def comput_new():
    # caculate score of dice
    truth_path = '/home/gf/guofeng/all_120/rawdata/one/'
    output_path = '/home/data_new/guofeng/projects/Segmentation/NPCCode/two_seg_new/score/'
    predict_path = '/home/gf/guofeng/all_120/rawdata/another/'
    score_bin_dir = r"/home/gf/guofeng/ImageValidationISLES/bin"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    #tpo = scorer.create_t_p_o_list(truth_path, predict_path, output_path)
    t_p_o_list = []
    predict_list = os.listdir(predict_path)
    #print(predict_list)
    truth_list = os.listdir(truth_path)
    truth_seg_tail = "T1SegRAIROIResampling.mha"
    pred_seg_tail = 'T1SegRAIROIResampling.mha'
    for pred in predict_list:
        #for truth in truth_list:
            #if pred == truth:
        name = pred.split('.')[0]
        t = truth_path + pred
        p = predict_path + pred
        o = output_path + name + ".json"
        t_p_o_list.append((t, p, o))
    print(t_p_o_list)
    os.chdir(score_bin_dir)
    for i in t_p_o_list:
        truth = i[0]
        predict = i[1]
        output = i[2]
        os.system("./ImageValidationISLES " + truth + " " + predict + " " + output)
#comput_new()
def find_best_result():
    score_path = './result/comp_result/xception_aspp_softmax_jacc/score'


