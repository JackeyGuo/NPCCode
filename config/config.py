# model parameter
learn_rate = 0.001
lr_decay_step = 500
lr_decay_rate = 0.9
iter_step = 5000
#record_step=100
# softmax or sigmoid
predict_op = 'softmax'
save_interval = 500
# if need to manually run model set Ture
one_by_one = False
# 1 2 3 4 5
begin_fold_iter = 5
end_fold_iter = 5
fold_num = 5
# model name [cnn_v1,cnn_v2, denseDilatedASPP, unet, deeplab, unet_3d, mynet,
# resnet_aspp, xception_aspp, deform_xception_aspp, resnet_aspp_5fold,vnet,vnet_new]
name = 'xception_aspp'
use_weight = False
use_sqrt_weights = False
use_dst_weight = False
use_focal_loss = False
only_use_dice = False
only_jacc_loss = True
use_param = False
jacc_entropy_loss = False
use_dice_loss = False
use_jacc_loss = False


block_size=4
block_count=4
weight_loss=0
use_bc=1

# data information
data_path = '/home/data_new/NPC/all_120/all_resized_124'
weight_distance_path = '/home/data_new/NPC/all_120/all_resized_124_dstmap'
# original patient data
patient_file_tail = 'T1RAIROIResamplingNormalize.mha'
# segment data
patient_seg_tail = 'T1SegRAIROIResampling.mha'
patient_weight_tail = 'dst.mha'

predict_tail = '.mha'
if one_by_one == True:
    output_path = '/home/data_new/guofeng/projects/Segmentation/NPCCode/result/%s_in%s/'%(name,begin_fold_iter)
else:
    output_path = '/home/data_new/guofeng/projects/Segmentation/NPCCode/result/%s_noaxmap/' % name
#leave_step = 25
#leave_step = 20

# GPUs
gpu = '7'
