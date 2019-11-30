import tensorflow as tf
import numpy as np
import os
import SimpleITK as sitk
import config.config as cfg

data_path = cfg.data_path
patient_tail = cfg.predict_tail
pat = './result/cnn_v2/predict/BAI_HUI_0014985428_'

def BilinearUpsample3d(inpt, up_factor):
    #inpt = tf.squeeze(inpt)
    inpt = tf.convert_to_tensor(inpt)
    print(inpt.get_shape())
    # inpt_height, inpt_width, inpt_depth, n_inpt = inpt.get_shape().to_list()
    inpt_depth, inpt_height, inpt_width, n_inpt= [int(d) for d in inpt.get_shape()]

    output_height = up_factor * inpt_height
    output_width = up_factor * inpt_width
    output_depth = up_factor * inpt_depth
    n_output = n_inpt

    # inpt = tf.reshape(inpt, (inpt_depth, n_inpt, inpt_height, inpt_width))
    # inpt = np.transpose(inpt, (2, 3, 0, 1))
    # [batch, height, width, channels]
    pre_res = tf.image.resize_images(inpt, [output_height, output_width])
    shuffle_res = tf.transpose(pre_res, (1, 0, 3, 2))

    res = tf.image.resize_images(shuffle_res, [output_depth, n_output])
    res = tf.transpose(res, (1, 0, 3, 2))

    print(res)
    return res

#train_path = os.path.join(data_path, pat + patient_tail)
train_path = os.path.join(data_path, pat)
pat_arr = sitk.GetArrayFromImage(sitk.ReadImage(train_path))
pat_arr = pat_arr[:,:,:,np.newaxis]


#BilinearUpsample3d(pat_arr,2)