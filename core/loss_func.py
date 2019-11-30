import tensorflow as tf
import config.config_dst as cfg_dst
import config.config as cfg
import tensorlayer as tl
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import numpy as np

''' Loss Function Definition'''


'''Dice Loss, depth 2'''

def dice_loss_function(prediction, ground_truth):
    ground_truth = tf.one_hot(indices=ground_truth, depth=1)
    dice = 0
    prediction, ground_truth = convert_type(logits=prediction, labels=ground_truth)
    for i in range(2):
        # reduce_mean calculation
        intersection = tf.reduce_mean(prediction * ground_truth)
        union_prediction = tf.reduce_sum(prediction * prediction)
        union_ground_truth = tf.reduce_sum(ground_truth * ground_truth)
        union = union_ground_truth + union_prediction
        dice = dice + 1 - 2 * intersection / union
    return dice

'''SoftMax Loss'''

def softmax_loss_function(logits, labels, dst_weight):
    # loss = weighted * - target * log(softmax(logits))
    # weighted?
    softmax_prediction = tf.nn.softmax(logits=logits)
    ground_truth = tf.one_hot(indices=labels, depth=2)
    softmax_prediction, ground_truth = convert_type(softmax_prediction, ground_truth)
    use_weight = cfg.use_weight
    use_dst_weight = cfg.use_dst_weight
    if use_dst_weight == True and dst_weight is not None:
        # pw = (1 - tf.reduce_mean(labels_slice)) / tf.reduce_mean(labels_slice)
        dst_weight = ops.convert_to_tensor(dst_weight)
        pw = dst_weight
        loss = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=softmax_prediction, targets=ground_truth, pos_weight=pw))
        # change it to calc weight.
    elif use_weight == True:
        #pw = (1 - tf.reduce_mean(ground_truth)) / tf.reduce_mean(ground_truth)
        #loss = tf.reduce_mean(
            #tf.nn.weighted_cross_entropy_with_logits(logits=softmax_prediction, targets=ground_truth, pos_weight=pw))
        loss = 0
        for i in range(2):
            class_i_ground_truth = ground_truth[:, :, :, :, i]
            class_i_prediction = softmax_prediction[:, :, :, :, i]
            #weighted = 1 - (tf.reduce_sum(class_i_ground_truth) / tf.reduce_sum(ground_truth))
            if cfg.use_sqrt_weights == True:
                weighted = tf.sqrt((1 - tf.reduce_mean(class_i_ground_truth)) / tf.reduce_mean(class_i_ground_truth))
            else:
                weighted = tf.divide((1 - tf.reduce_mean(class_i_ground_truth)) / tf.reduce_mean(class_i_ground_truth),4)
            loss = loss - tf.reduce_mean(
                weighted * class_i_ground_truth *tf.log(tf.clip_by_value(class_i_prediction, 1e-8, 1.0)) +
                    (1 - class_i_ground_truth) * tf.log(tf.clip_by_value(1-class_i_prediction, 1e-8, 1.0)))
                # Clips tensor values to a specified min and max.
            tf.summary.scalar("pos_weight", tf.reduce_mean(weighted))
    else:
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    return loss

'''Cross_entropy with wights loss'''
def entropy_loss_function(logits,labels,dst_weight=None):
    # no need to do this active operation
    #sigmoid_prediction = tf.nn.sigmoid(logits)

    logits_slice,labels_slice = convert_type(logits,labels)

    use_weight = cfg.use_weight
    use_dst_weight = cfg.use_dst_weight
    if use_dst_weight == True and dst_weight is not None:
        dst_weight = ops.convert_to_tensor(dst_weight)
        pw = dst_weight
        tf.summary.scalar("pos_weight", tf.reduce_mean(pw))
        loss = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=logits_slice, targets=labels_slice, pos_weight=pw))
        # change it to calc weight.
    elif use_weight == True:
        pw = (1 - tf.reduce_mean(labels_slice)) / tf.reduce_mean(labels_slice)
        #weighted = 1 - (tf.reduce_sum(class_i_ground_truth) / tf.reduce_sum(ground_truth))
        tf.summary.scalar("pos_weight", tf.reduce_mean(pw))
        loss = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=logits_slice, targets=labels_slice, pos_weight=pw))
    else:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_slice,labels=labels_slice))

    return loss

def convert_type(logits,labels):

    logits = tf.cast(logits, dtype=tf.float32)
    labels = tf.cast(labels, dtype=tf.float32)
    #logits = tf.squeeze(logits,axis=-1)
    #labels_slice = tf.squeeze(labels,axis=-1)

    return logits,labels

def tensorlayer_dice_loss(logits,labels):
    # sigmoid activation function
    logits = tf.nn.softmax(logits)
    labels = tf.one_hot(indices=labels, depth=2)
    #print(labels)
    # convert logits and labels to type-float32, and squeeze to three dimention
    logits, labels = convert_type(logits,labels)
    total_dice = 0
    total_dice = 1 - tl.cost.dice_coe(logits,labels)
    '''for i in range(2):
        dice_loss = 1 - tl.cost.dice_coe(logits[:, :, :, :, i],labels[:, :, :, :, i])
        total_dice += dice_loss'''
    #iou_loss = 1 - tl.cost.iou_coe(outputs,labels)
    #dice_hard = 1 - tl.cost.dice_hard_coe(outputs,labels)
    return total_dice

def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """

    prediction_tensor, target_tensor = convert_type(prediction_tensor, target_tensor)

    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    #sigmoid_p = prediction_tensor
    # 新建一个与给定的tensor类型大小一致的tensor，其所有元素为0
    zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    #tf.where(input, a,b)，其中a，b均为尺寸一致的tensor，作用是将a中对应input中true的位置的元素值不变，
    #其余元素进行替换，替换成b中对应位置的元素值

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = tf.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = tf.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    '''pt_1 = tf.where(tf.equal(target_tensor, 1), sigmoid_p, tf.ones_like(target_tensor))
    pt_0 = tf.where(tf.equal(target_tensor, 0), sigmoid_p, tf.zeros_like(sigmoid_p))

    per_entry_cross_ent = - alpha * ((1 - pt_1) ** gamma) * tf.log(tf.clip_by_value(pt_1, 1e-8, 1.0)) \
                          - (1 - alpha) * (pt_0 ** gamma) * tf.log(tf.clip_by_value(1.0 - pt_0, 1e-8, 1.0))
    per_entry_cross_ent = -np.sum(alpha * np.power(1. - pt_1, gamma) * tf.log(pt_1)) - np.sum(
        (1 - alpha) * np.power(pt_0, gamma) * tf.log(1. - pt_0))'''
    return tf.reduce_sum(per_entry_cross_ent)

def jacc_loss(output, target, loss_type='jaccard', axis=[1,2,3], smooth=1e-5):

    output = tf.nn.softmax(output)
    target = tf.one_hot(indices=target, depth=2)
    # convert logits and labels to type-float32, and squeeze to three dimention
    output, target = convert_type(output, target)

    total_jacc = 0
    for i in range(2):
        inse = tf.reduce_sum(output[:, :, :, :, i] * target[:, :, :, :, i], axis=axis)
        if loss_type == 'jaccard':
            l = tf.reduce_sum(output[:, :, :, :, i] * output[:, :, :, :, i], axis=axis)
            r = tf.reduce_sum(target[:, :, :, :, i] * target[:, :, :, :, i], axis=axis)
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(output[:, :, :, :, i], axis=axis)
            r = tf.reduce_sum(target[:, :, :, :, i], axis=axis)
        else:
            raise Exception("Unknow loss_type")

        jacc = (inse + smooth) / (l + r -inse + smooth)
        jacc = 1 - tf.reduce_mean(jacc)

        total_jacc += jacc
    return total_jacc
def jacc_loss_new(output, target, loss_type='jaccard', axis=[1,2,3], smooth=1e-5):

    output = tf.nn.softmax(output)
    target = tf.one_hot(indices=target, depth=2)
    # convert logits and labels to type-float32, and squeeze to three dimention
    output, target = convert_type(output, target)

    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")

    jacc = (inse + smooth) / (l + r -inse + smooth)
    jacc = 1 - tf.reduce_mean(jacc)

    return jacc
def param_jacc_loss(output, target, param, loss_type='jaccard', axis=[1,2,3], smooth=1e-5):
    output = tf.nn.softmax(output)
    target = tf.one_hot(indices=target, depth=2)
    # convert logits and labels to type-float32, and squeeze to three dimention
    output, target = convert_type(output, target)

    #para = tf.reduce_sum(output * target * np.abs(param), axis=[1, 2, 3])
    inse = tf.reduce_sum(output * target, axis=[1, 2, 3])
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    jacc = (inse + smooth) / (l + r - inse + smooth)

    jacc = np.abs(tf.reduce_mean(param))*(1 - tf.reduce_mean(jacc))

    return jacc