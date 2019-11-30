import tensorflow as tf
import numpy as np
from core.deformable_conv3d import *

def deform_conv2d(inputs, num_outputs, kernel_size, scope, norm=True, d_format='NHWC'):
    # 生成offset-field
    offset = tf.contrib.layers.conv2d(
        inputs, kernel_size[0] * kernel_size[0] * 2, [3, 3], scope=scope + '/offset',
        data_format=d_format, activation_fn=tf.nn.tanh, weights_initializer=tf.zeros_initializer(dtype=tf.float32),
        biases_initializer=None)

    # 生成deformed feature
    input_shape = [inputs.shape[0].value, inputs.shape[1].value, inputs.shape[2].value, inputs.shape[3].value]
    dcn = DCN(input_shape, kernel_size)
    deformed_feature = dcn.deform_conv(inputs, offset, scope)

    # 完成卷积操作
    outputs = tf.contrib.layers.conv2d(
        deformed_feature, num_outputs, kernel_size, scope=scope,
        stride=kernel_size[0], padding="VALID", data_format=d_format,
        activation_fn=None, biases_initializer=None)

    if norm:
        outputs = tf.contrib.layers.batch_norm(
            outputs, decay=0.9, center=True, activation_fn=tf.nn.relu,
            updates_collections=None, epsilon=1e-5, scope=scope + '/batch_norm',
            data_format=d_format)
    else:
        outputs = tf.nn.relu(outputs, name=scope + '/relu')
    return outputs

def deform_conv3d(inputs, num_outputs, kernel_size, scope, norm=True, d_format='NHWC'):
    # 生成offset-field
    offset = tf.contrib.layers.conv3d(
        inputs, kernel_size[0] * kernel_size[0] * 2, [3, 3, 3], scope=scope + '/offset',
        data_format=d_format, activation_fn=tf.nn.tanh, weights_initializer=tf.zeros_initializer(dtype=tf.float32),
        biases_initializer=None)

    # 生成deformed feature
    input_shape = [inputs.shape[0].value, inputs.shape[1].value, inputs.shape[2].value, inputs.shape[3].value]
    dcn = DCN(input_shape, kernel_size)
    deformed_feature = dcn.deform_conv(inputs, offset, scope)

    # 完成卷积操作
    outputs = tf.contrib.layers.conv3d(
        deformed_feature, num_outputs, kernel_size, scope=scope,
        stride=kernel_size[0], padding="VALID", data_format=d_format,
        activation_fn=None, biases_initializer=None)

    if norm:
        outputs = tf.contrib.layers.batch_norm(
            outputs, decay=0.9, center=True, activation_fn=tf.nn.relu,
            updates_collections=None, epsilon=1e-5, scope=scope + '/batch_norm',
            data_format=d_format)
    else:
        outputs = tf.nn.relu(outputs, name=scope + '/relu')
    return outputs