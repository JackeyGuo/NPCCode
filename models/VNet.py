from core.layers import convolution_3d, deconvolution_3d, prelu, _conv3d, _deconv3d, v_conv3d
from core.loss_func import *

def convolution_block(layer_input, n_channels, num_convolutions):
    x = layer_input
    for i in range(num_convolutions - 1):
        with tf.variable_scope('conv_' + str(i+1)):
            #x = convolution_3d(x, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1])
            x = _conv3d(x,output_feature=n_channels,kernel_size=5,stride=1)
            #x = v_conv3d(x,kernel_shape=[5, 5, 5, n_channels, n_channels],stride=1)
            x = prelu(x)
    #x = convolution_3d(x, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1])
    #x = v_conv3d(x, kernel_shape=[5, 5, 5, n_channels, n_channels], stride=1)
    x = _conv3d(x, output_feature=n_channels, kernel_size=5, stride=1)
    x = x + layer_input
    return prelu(x)

def convolution_block_2(layer_input, fine_grained_features, n_channels, num_convolutions):

    x = tf.concat((layer_input, fine_grained_features), axis=-1)

    with tf.variable_scope('conv_' + str(1)):
        #x = convolution_3d(x, [5, 5, 5, n_channels * 2, n_channels], [1, 1, 1, 1, 1])
        #x = v_conv3d(x,kernel_shape=[5, 5, 5, n_channels * 2, n_channels],stride=1)
        x = _conv3d(x, output_feature=n_channels, kernel_size=5, stride=1)
    for i in range(1, num_convolutions - 1):
        with tf.variable_scope('conv_' + str(i+1)):
            #x = convolution_3d(x, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1])
            x = _conv3d(x, output_feature=n_channels, kernel_size=5, stride=1)
            #x = v_conv3d(x, kernel_shape=[5, 5, 5, n_channels, n_channels], stride=1)
            x = prelu(x)
    #x = convolution_3d(x, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1])
    x = _conv3d(x, output_feature=n_channels, kernel_size=5, stride=1)
    #x = v_conv3d(x, kernel_shape=[5, 5, 5, n_channels, n_channels], stride=1)
    x = x + layer_input
    return prelu(x)

def down_convolution(layer_input, in_channels):
    with tf.variable_scope('down_convolution'):
        #x = convolution_3d(layer_input, [2, 2, 2, in_channels, in_channels * 2], [1, 2, 2, 2, 1])
        x = _conv3d(layer_input, output_feature=in_channels * 2, kernel_size=2, stride=2)
        #x = v_conv3d(layer_input, kernel_shape=[2, 2, 2, in_channels, in_channels * 2], stride=2)
        return prelu(x)

def up_convolution(layer_input, output_shape, in_channels):
    with tf.variable_scope('up_convolution'):
        #x = deconvolution_3d(layer_input, [2, 2, 2, in_channels // 2, in_channels], output_shape, [1, 2, 2, 2, 1])
        x = _deconv3d(layer_input,input_shape=[2, 2, 2, in_channels // 2, in_channels],
                      output_shape=output_shape,stride=2)
        return prelu(x)

class V_Net(object):
    def __init__(self, is_training):
        self.is_training = is_training

    def v_net(tf_input, input_channels, output_channels=1, n_channels=16):

        with tf.variable_scope('contracting_path'):

            # if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input channel
            if input_channels == 1:
                c0 = tf.tile(tf_input, [1, 1, 1, 1, n_channels])
            else:
                with tf.variable_scope('level_0'):
                    c0 = convolution_3d(tf_input, [5, 5, 5, input_channels, n_channels], [1, 1, 1, 1, 1])
                    c0 = prelu(c0)

            with tf.variable_scope('level_1'):
                c1 = convolution_block(c0, n_channels, 1)
                c12 = down_convolution(c1, n_channels)

            with tf.variable_scope('level_2'):
                c2 = convolution_block(c12, n_channels * 2, 2)
                c22 = down_convolution(c2, n_channels * 2)

            with tf.variable_scope('level_3'):
                c3 = convolution_block(c22, n_channels * 4, 3)
                c32 = down_convolution(c3, n_channels * 4)

            with tf.variable_scope('level_4'):
                c4 = convolution_block(c32, n_channels * 8, 3)
                c42 = down_convolution(c4, n_channels * 8)

            with tf.variable_scope('level_5'):
                c5 = convolution_block(c42, n_channels * 16, 3)
                c52 = up_convolution(c5, tf.shape(c4), n_channels * 16)

        with tf.variable_scope('expanding_path'):

            with tf.variable_scope('level_4'):
                e4 = convolution_block_2(c52, c4, n_channels * 8, 3)
                e42 = up_convolution(e4, tf.shape(c3), n_channels * 8)

            with tf.variable_scope('level_3'):
                e3 = convolution_block_2(e42, c3, n_channels * 4, 3)
                e32 = up_convolution(e3, tf.shape(c2), n_channels * 4)

            with tf.variable_scope('level_2'):
                e2 = convolution_block_2(e32, c2, n_channels * 2, 2)
                e22 = up_convolution(e2, tf.shape(c1), n_channels * 2)

            with tf.variable_scope('level_1'):
                e1 = convolution_block_2(e22, c1, n_channels, 1)
                with tf.variable_scope('output_layer'):
                    logits = convolution_3d(e1, [1, 1, 1, n_channels, output_channels], [1, 1, 1, 1, 1])

        return logits

    def inference_op(self, _input):
        input_channels = 1
        output_channels = 2
        n_channels = 16
        _input = tf.pad(_input, np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0]]), name='pad_1')

        with tf.variable_scope('contracting_path'):

            # if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input channel
            if input_channels == 1:
                c0 = tf.tile(_input, [1, 1, 1, 1, n_channels])
            else:
                with tf.variable_scope('level_0'):
                    #c0 = convolution_3d(_input, [5, 5, 5, input_channels, n_channels], [1, 1, 1, 1, 1])
                    c0 = _conv3d(_input, output_feature=n_channels, kernel_size=5, stride=1)
                    #c0 = v_conv3d(_input, kernel_shape=[5, 5, 5, input_channels, n_channels], stride=1)
                    c0 = prelu(c0)
            print(c0)
            with tf.variable_scope('level_1'):
                c1 = convolution_block(c0, n_channels, 1)
                c12 = down_convolution(c1, n_channels)

            with tf.variable_scope('level_2'):
                c2 = convolution_block(c12, n_channels * 2, 2)
                c22 = down_convolution(c2, n_channels * 2)

            with tf.variable_scope('level_3'):
                c3 = convolution_block(c22, n_channels * 4, 3)
                c32 = down_convolution(c3, n_channels * 4)

            with tf.variable_scope('level_4'):
                c4 = convolution_block(c32, n_channels * 8, 3)
                c42 = down_convolution(c4, n_channels * 8)

            with tf.variable_scope('level_5'):
                c5 = convolution_block(c42, n_channels * 16, 3)
                c52 = up_convolution(c5, tf.shape(c4), n_channels * 16)

        with tf.variable_scope('expanding_path'):

            with tf.variable_scope('level_4'):
                e4 = convolution_block_2(c52, c4, n_channels * 8, 3)
                e42 = up_convolution(e4, tf.shape(c3), n_channels * 8)

            with tf.variable_scope('level_3'):
                e3 = convolution_block_2(e42, c3, n_channels * 4, 3)
                e32 = up_convolution(e3, tf.shape(c2), n_channels * 4)

            with tf.variable_scope('level_2'):
                e2 = convolution_block_2(e32, c2, n_channels * 2, 2)
                e22 = up_convolution(e2, tf.shape(c1), n_channels * 2)

            with tf.variable_scope('level_1'):
                e1 = convolution_block_2(e22, c1, n_channels, 1)
                with tf.variable_scope('output_layer'):
                    #logits = convolution_3d(e1, [1, 1, 1, n_channels, output_channels], [1, 1, 1, 1, 1])
                    logits = _conv3d(e1, output_feature=output_channels, kernel_size=1, stride=1)
                    #logits = v_conv3d(e1, kernel_shape=[1, 1, 1, n_channels, output_channels], stride=1)
                    logits = logits[:, :103, :198, :]
        with tf.variable_scope('prediction'):
            softmax_prob = tf.nn.softmax(logits=logits, name='softmax_prob')
            predicted_label = tf.argmax(input=softmax_prob, axis=4, name='predicted_label')
            #predicted_label = tf.nn.sigmoid(logits, name='predicted_label')
        #print(logits,predicted_label)
        return logits,predicted_label

    def loss_op(self, logits, labels, dst_weight=None):

        with tf.name_scope("weighted_cross_entropy"):
            if cfg.use_dice_loss == True:
                self.total_loss = tensorlayer_dice_loss(logits[0], labels)
            if cfg.use_jacc_loss == True:
                self.total_loss = jacc_loss_new(logits[0], labels)
            else:
                self.total_loss = softmax_loss_function(logits[0], labels, dst_weight)
        with tf.name_scope("accuracy"):
            if cfg.predict_op == 'sigmoid':
                correct_pred = tf.equal(tf.squeeze(tf.cast(logits[1] + 0.5, tf.int32),axis=-1), tf.cast(labels,dtype=tf.int32))
            else:
                #correct_pred = tf.equal(logits[1],tf.cast(labels, dtype=tf.int64))
                correct_pred = tf.reduce_sum(logits[1]) / tf.reduce_sum(tf.cast(labels, dtype=tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        #tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar("total_loss", self.total_loss)

        return self.total_loss, accuracy