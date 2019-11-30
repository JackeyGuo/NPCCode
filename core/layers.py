import tensorflow as tf
import numpy as np
import tflearn
from core.deformable_conv3d import DCN

slim = tf.contrib.slim

def _weight_variable_msra(shape, name):
    return tf.get_variable(
        name=name,
        shape=shape,
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
#initializer=tf.contrib.layers.variance_scaling_initializer())
# 3D convolution
def _conv3d(inputs, output_feature, kernel_size, stride, dilation_rate=1, padding='same', use_bias=False, name='conv'):

    tensor = tf.layers.conv3d(
        inputs=inputs,                  # Tensor input
        filters=output_feature,        # Integer, the dimensionality of the output space
        kernel_size=kernel_size,        # An integer or tuple/list of 3, depth, height and width
        strides=stride,                 # (1, 1, 1)
        padding=padding,                # "valid" or "same", same: zero padding
        data_format='channels_last',    # channels_last (batch, depth, height, width, channels)
                                        # channels_first (batch, channels, depth, height, width)
        dilation_rate=dilation_rate,        # incompatible problem with stride value != 1
        activation=None,                # None to maintain a linear activation
        use_bias=use_bias,
        kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
        kernel_regularizer=slim.l2_regularizer(scale=0.0005),
        bias_initializer=tf.zeros_initializer(),
        name=name,                      # the name of the layer
    )

    return tensor

def _batch_norm(_input):

    output = tf.contrib.layers.batch_norm(
        _input, 0.99, scale=True,updates_collections=None)
    return output

def conv3d(_input, kernel_size, stride, output_feature, padding="SAME"):
    in_features = int(_input.get_shape()[-1])
    kernel = _weight_variable_msra(
        [kernel_size, kernel_size, kernel_size, in_features, output_feature],
        name='kernel')
    strides = [1, stride, stride, stride, 1]
    output = tf.nn.conv3d(_input, kernel, strides, padding)
    return output

def _conv3d_atrous(_input, kernel_size=3, stride=1, atrous=1, output_feature=8, padding="SAME"):
    in_features = int(_input.get_shape()[-1])
    kernel = _weight_variable_msra(
        [kernel_size, kernel_size, kernel_size, in_features, output_feature],
        name='kernel')
    strides = [stride, stride, stride]
    atrous_size = [atrous, atrous, atrous]
    return tf.nn.convolution(_input, kernel, padding, strides, atrous_size)

def _create_conv_layer(_input, kernel_size=3, stride=1, atrous=1, output_feature=16, padding="SAME"):
    _input = _batch_norm(_input)
    _input = tf.nn.relu(_input)
    if atrous == 1:
        output = conv3d(_input, kernel_size, stride, output_feature, padding)
    else:
        output = _conv3d_atrous(_input, kernel_size, stride, atrous, output_feature, padding)
    return output

def _create_dconv_layer(_input, kernel_size=3, stride=1):
    in_channel = int(_input.get_shape()[-1])
    kernel = _weight_variable_msra(
        [kernel_size, kernel_size, kernel_size, in_channel, in_channel], "dconv_kernel")
    strides = [1, stride, stride, stride, 1]
    #output_shape = [int(img_shape[0]), int(img_shape[1]), int(img_shape[2]), int(img_shape[3]), 32]
    out_size = _input.get_shape()
    output_shape = [1, int(out_size[1])*2, int(out_size[2]*2), int(out_size[3]*2), in_channel]
    output = tf.nn.conv3d_transpose(_input, kernel, output_shape, strides, padding='SAME', data_format='NDHWC')

    return output

def _deconv3d(inputs, input_shape, output_shape, stride=1, name='deconv'):
    # depth, height and width
    batch, in_depth, in_height, in_width, _ = [int(d) for d in inputs.get_shape()]
    #output_channels = int(inputs.get_shape()[-1])

    dev_filter = tf.get_variable(
        name=name+'/filter',          # name of the new or existing variable
        shape=input_shape,
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
        regularizer=slim.l2_regularizer(scale=0.0005),
    )
    deconv = tf.nn.conv3d_transpose(
        value=inputs,                   # [batch, depth, height, width, in_channels]
        filter=dev_filter,              # [depth, height, width, output_channels, in_channels]
        output_shape=output_shape,
        strides=[1, stride, stride, stride, 1],
        padding='SAME',
        data_format='NDHWC',
        name=name
    )
    return deconv

# 3D Deconvolution
def deconv3d(inputs, output_channels, name='deconv'):
    # depth, height and width
    batch, in_depth, in_height, in_width, in_channels = [int(d) for d in inputs.get_shape()]
    dev_filter = tf.get_variable(
        name=name+'/filter',          # name of the new or existing variable
        shape=[3, 3, 3, output_channels, in_channels],
        # 4, 4, 4, depth, height and width
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
        regularizer=slim.l2_regularizer(scale=0.0005),
    )
    deconv = tf.nn.conv3d_transpose(
        value=inputs,                   # [batch, depth, height, width, in_channels]
        filter=dev_filter,              # [depth, height, width, output_channels, in_channels]
        output_shape=[batch, in_depth*2, in_height*2, in_width*2, output_channels],
        strides=[1, 2, 2, 2, 1],
        padding='SAME',
        data_format='NDHWC',
        name=name
    )
    '''Strides and Filter shape, draw the pictures'''
    return deconv


def _conv_layer(_input, kernel_size=3, stride=1, atrous=1, output_feature=16, padding="SAME"):

    if atrous == 1:
        output = _conv3d(_input, kernel_size, stride, output_feature, padding)
        print('output1:',output)
    else:
        output = _conv3d_atrous(_input, kernel_size, stride, atrous, output_feature, padding)
    output = _active_layer(output)
    return output

def conv_bn_prelu(inputs, output_channels, kernel_size, stride, is_training, name,
                 padding='same', use_bias=False):
    with tf.variable_scope(name_or_scope=name):
        conv = _conv3d(inputs, output_channels, kernel_size, stride, padding=padding,
                      use_bias=use_bias, name=name+'_conv')
        '''device control?'''
        bn = tf.contrib.layers.batch_norm(
            inputs=conv,                # tensor, first dimension of batch_size
            decay=0.9,                  # recommend trying decay=0.9
            scale=True,                 # If True, multiply by gamma. If False, gamma is not used
            epsilon=1e-5,               # Small float added to variance to avoid dividing by zero
            updates_collections=None,   # tf.GraphKeys.UPDATE_OPS,
            # updates_collections: Collections to collect the update ops for computation.
            # The updates_ops need to be executed with the train_op.
            # If None, a control dependency would be added to make sure the updates are computed in place
            is_training=is_training,
            # In training mode it would accumulate the statistics of the moments into moving_mean
            # and moving_variance using an exponential moving average with the given decay.
            scope=name+'_batch_norm',         # variable_scope

        )
        prelu = tflearn.prelu(bn, name=name+'_prelu')
    return prelu
def bn_prelu_conv(inputs, output_channels, kernel_size, stride, is_training, name,
                 padding='same', use_bias=False):
    with tf.variable_scope(name_or_scope=name):
        '''device control?'''
        bn = tf.contrib.layers.batch_norm(
            inputs=inputs,                # tensor, first dimension of batch_size
            decay=0.9,                  # recommend trying decay=0.9
            scale=True,                 # If True, multiply by gamma. If False, gamma is not used
            epsilon=1e-5,               # Small float added to variance to avoid dividing by zero
            updates_collections=None,   # tf.GraphKeys.UPDATE_OPS,
            # updates_collections: Collections to collect the update ops for computation.
            # The updates_ops need to be executed with the train_op.
            # If None, a control dependency would be added to make sure the updates are computed in place
            is_training=is_training,
            # In training mode it would accumulate the statistics of the moments into moving_mean
            # and moving_variance using an exponential moving average with the given decay.
            scope=name+'_batch_norm',         # variable_scope

        )
        prelu = tflearn.prelu(bn, name=name+'_prelu')
        conv = _conv3d(prelu, output_channels, kernel_size, stride, padding=padding,
                       use_bias=use_bias, name=name + '_conv')
    return conv

def atrous_bn_prelu(inputs, output_channels, kernel_size, stride, is_training, name, dilation_rate,
                 padding='same', use_bias=False):
    with tf.variable_scope(name_or_scope=name):
        conv = _conv3d(inputs, output_channels, kernel_size, stride, padding=padding, dilation_rate=dilation_rate,
                      use_bias=use_bias, name=name+'_conv')
        '''device control?'''
        bn = tf.contrib.layers.batch_norm(
            inputs=conv,                # tensor, first dimension of batch_size
            decay=0.9,                  # recommend trying decay=0.9
            scale=True,                 # If True, multiply by gamma. If False, gamma is not used
            epsilon=1e-5,               # Small float added to variance to avoid dividing by zero
            updates_collections=None,   # tf.GraphKeys.UPDATE_OPS,
            # updates_collections: Collections to collect the update ops for computation.
            # The updates_ops need to be executed with the train_op.
            # If None, a control dependency would be added to make sure the updates are computed in place
            is_training=is_training,
            # In training mode it would accumulate the statistics of the moments into moving_mean
            # and moving_variance using an exponential moving average with the given decay.
            scope=name+'_batch_norm',         # variable_scope

        )
        prelu = tflearn.prelu(bn, name=name+'_prelu')
    return prelu

# 3D Deconvolution, Batch normalization, ReLU unit
def deconv_bn_prelu(inputs, output_channels, is_training, name):
    with tf.variable_scope(name):
        deconv = deconv3d(inputs=inputs, output_channels=output_channels, name=name+'_deconv')
        '''device control?'''
        bn = tf.contrib.layers.batch_norm(inputs=deconv, decay=0.9, scale=True, epsilon=1e-5,
                                          updates_collections=None, is_training=is_training,
                                          scope=name+'_batch_norm')
        prelu = tflearn.prelu(bn, name=name+'_prelu')
    return prelu
def bn_prelu_deconv(inputs, output_channels, is_training, name):
    with tf.variable_scope(name):
        '''device control?'''
        bn = tf.contrib.layers.batch_norm(inputs=inputs, decay=0.9, scale=True, epsilon=1e-5,
                                          updates_collections=None, is_training=is_training,
                                          scope=name+'_batch_norm')
        prelu = tflearn.prelu(bn, name=name+'_prelu')
        deconv = deconv3d(inputs=prelu, output_channels=output_channels, name=name + '_deconv')
    return deconv

def _active_layer(_input,name,is_training):

    bn = tf.contrib.layers.batch_norm(
        inputs=_input,  # tensor, first dimension of batch_size
        decay=0.9,  # recommend trying decay=0.9
        scale=True,  # If True, multiply by gamma. If False, gamma is not used
        epsilon=1e-5,  # Small float added to variance to avoid dividing by zero
        updates_collections=None,  # tf.GraphKeys.UPDATE_OPS,
        # updates_collections: Collections to collect the update ops for computation.
        # The updates_ops need to be executed with the train_op.
        # If None, a control dependency would be added to make sure the updates are computed in place
        is_training=is_training,
        # In training mode it would accumulate the statistics of the moments into moving_mean
        # and moving_variance using an exponential moving average with the given decay.
        scope=name + '_batch_norm',  # variable_scope
    )
    prelu = tflearn.prelu(bn, name=name + '_prelu')

    return prelu

def deform_conv3d(inputs, output_channels, kernel_size, name, norm=True, d_format='NHWC'):
    num_outputs = kernel_size[0] * kernel_size[1] * kernel_size[2] * 3

    # 生成offset-field
    offset = tf.contrib.layers.conv3d(
        inputs, num_outputs, [3, 3, 3], scope=name + '/offset',
        data_format=d_format, activation_fn=tf.nn.tanh, weights_initializer=tf.zeros_initializer(dtype=tf.float32),
        biases_initializer=None)

    # 生成deformed feature
    input_shape = [inputs.shape[0].value, inputs.shape[1].value, inputs.shape[2].value, inputs.shape[3].value, inputs.shape[4].value]
    dcn = DCN(input_shape, kernel_size)
    deformed_feature = dcn.deform_conv(inputs, offset, name)

    # 完成卷积操作
    outputs = tf.contrib.layers.conv3d(
        deformed_feature, output_channels, kernel_size, scope=name,
        stride=kernel_size[0], padding="VALID", data_format=d_format,
        activation_fn=None, biases_initializer=None)

    if norm:
        outputs = tf.contrib.layers.batch_norm(
            outputs, decay=0.9, center=True, activation_fn=tf.nn.relu,
            updates_collections=None, epsilon=1e-5, scope=name + '/batch_norm',
            data_format=d_format)
    else:
        outputs = tf.nn.relu(outputs, name=name + '/relu')
    return outputs

def BilinearUpsample3d(inpt, up_factor):
    inpt = tf.squeeze(inpt)

    #inpt_height, inpt_width, inpt_depth, n_inpt = inpt.get_shape().to_list()
    inpt_depth, inpt_height, inpt_width, n_inpt = [int(d) for d in inpt.get_shape()]

    output_height = up_factor * inpt_height
    output_width = up_factor * inpt_width
    output_depth = up_factor * inpt_depth
    n_output = n_inpt

    #inpt = tf.reshape(inpt, (inpt_depth, n_inpt, inpt_height, inpt_width))
    #inpt = np.transpose(inpt, (2, 3, 0, 1))
    # [batch, height, width, channels]
    pre_res = tf.image.resize_images(inpt, [output_height, output_width])
    shuffle_res = tf.transpose(pre_res,(1, 0, 3, 2))
    #print(pre_res,shuffle_res)
    res = tf.image.resize_images(shuffle_res, [output_depth, n_output])
    #print(res)
    res = tf.transpose(res, (1, 0, 3, 2))
    #print(res)
    return res

def crop_tensor(input_tensor, block_num):
    #print(input_tensor)
    data_shape = input_tensor.get_shape()
    #transed_shape = [int(data_shape[0]), int(data_shape[1]), int(data_shape[2]), int(data_shape[3])]
    central_x = int(data_shape[1]) // 2
    central_y = int(data_shape[2]) // 2
    central_z = int(data_shape[3]) // 2
    channles = int(data_shape[4])

    output_tensor = tf.slice(input_tensor, [0, central_x, central_y, central_z, 0],
                            [1, int(data_shape[1]) - central_x, int(data_shape[2]) - central_y,
                             int(data_shape[3]) - central_z, channles])
    #print(output_tensor)
    return output_tensor
# n_activations_prev_layer = patch_volume_prev * in_channels
# n_activations_current_layer = patch_volume * out_channels
# sqrt(3/(n_activations_prev_layer + n_activations_current_layer)) (assuming prev_patch==curr_patch)
def xavier_normal_dist_conv3d(shape):
    return tf.truncated_normal(shape, mean=0,
                               stddev=tf.sqrt(3. / (tf.reduce_prod(shape[:3]) * tf.reduce_sum(shape[3:]))))


def xavier_uniform_dist_conv3d(shape):
    with tf.variable_scope('xavier_glorot_initializer'):
        denominator = tf.cast((tf.reduce_prod(shape[:3]) * tf.reduce_sum(shape[3:])), tf.float32)
        lim = tf.sqrt(6. / denominator)
        return tf.random_uniform(shape, minval=-lim, maxval=lim)

# parametric leaky relu
def prelu(x):
    pre = tflearn.prelu(x)
    #alpha = tf.get_variable('alpha', shape=x.get_shape()[-1], dtype=x.dtype, initializer=tf.constant_initializer(0.1))
    #return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)
    return pre

def convolution_3d(layer_input, filter, strides, padding='SAME'):
    assert len(filter) == 5  # [filter_depth, filter_height, filter_width, in_channels, out_channels]
    assert len(strides) == 5  # must match input dimensions [batch, in_depth, in_height, in_width, in_channels]
    assert padding in ['VALID', 'SAME']

    w = tf.Variable(initial_value=xavier_uniform_dist_conv3d(shape=filter), name='weights')
    b = tf.Variable(tf.constant(1.0, shape=[filter[-1]]), name='biases')

    return tf.nn.conv3d(layer_input, w, strides, padding) + b

def v_conv3d(_input, kernel_shape, stride, padding="SAME"):
    #in_features = int(_input.get_shape()[-1])
    kernel = _weight_variable_msra(kernel_shape,name='kernel')
    strides = [1, stride, stride, stride, 1]
    output = tf.nn.conv3d(_input, kernel, strides, padding)
    return output


def deconvolution_3d(layer_input, filter, output_shape, strides, padding='SAME'):
    assert len(filter) == 5  # [depth, height, width, output_channels, in_channels]
    assert len(strides) == 5  # must match input dimensions [batch, depth, height, width, in_channels]
    assert padding in ['VALID', 'SAME']

    w = tf.Variable(initial_value=xavier_uniform_dist_conv3d(shape=filter), name='weights')
    b = tf.Variable(tf.constant(1.0, shape=[filter[-2]]), name='biases')

    return tf.nn.conv3d_transpose(layer_input, w, output_shape, strides, padding) + b


def max_pooling_3d(layer_input, ksize, strides, padding='SAME'):
    assert len(ksize) == 5  # [batch, depth, rows, cols, channels]
    assert len(strides) == 5  # [batch, depth, rows, cols, channels]
    assert ksize[0] == ksize[4]
    assert ksize[0] == 1
    assert strides[0] == strides[4]
    assert strides[0] == 1
    return tf.nn.max_pool3d(layer_input, ksize, strides, padding)