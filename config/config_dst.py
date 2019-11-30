
# model parameter
learn_rate = 0.001
lr_decay_step = 500
lr_decay_rate = 0.9
#iter_step=5000
iter_step = 5000
#record_step=100
predict_op = 'sigmoid'
save_interval = 500

# model name [cnn_v1,cnn_v2, denseDilatedASPP, unet, deeplab, unet_3d, mynet, mynet_focal]
name = 'mynet_focal'
use_weight = False
use_dst_weight = False
use_focal_loss = True

# data information
data_path = '/home/gf/guofeng/0120/all_resized_124'
weight_distance_path = '/home/gf/guofeng/0120/all_resized_124_dstmap'
# original patient data
patient_file_tail = 'T1RAIROIResamplingNormalize.mha'
# segment data
patient_seg_tail = 'T1SegRAIROIResampling.mha'
patient_weight_tail = 'dst.mha'

#output_path='/home/gf/guofeng/0120/NPCFrame/new_begin/test/provider'
predict_tail = '.mha'
output_path = './result/%s/'%name
leave_step = 25
#leave_step = 20

# GPUs
gpu = '1'