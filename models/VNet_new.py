from core.Layers import convolution_3d, deconvolution_3d, prelu
from core.loss_func import *

def convolution_block(layer_input, n_channels, num_convolutions):
    x = layer_input
    for i in range(num_convolutions - 1):
        with tf.variable_scope('conv_' + str(i+1)):
            x = convolution_3d(x, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1])
            x = prelu(x)
    x = convolution_3d(x, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1])
    x = x + layer_input
    return prelu(x)


def convolution_block_2(layer_input, fine_grained_features, n_channels, num_convolutions):

    x = tf.concat((layer_input, fine_grained_features), axis=-1)

    with tf.variable_scope('conv_' + str(1)):
        x = convolution_3d(x, [5, 5, 5, n_channels * 2, n_channels], [1, 1, 1, 1, 1])

    for i in range(1, num_convolutions - 1):
        with tf.variable_scope('conv_' + str(i+1)):
            x = convolution_3d(x, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1])
            x = prelu(x)

    x = convolution_3d(x, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1])
    x = x + layer_input
    return prelu(x)


def down_convolution(layer_input, in_channels):
    with tf.variable_scope('down_convolution'):
        x = convolution_3d(layer_input, [2, 2, 2, in_channels, in_channels * 2], [1, 2, 2, 2, 1])
        return prelu(x)


def up_convolution(layer_input, output_shape, in_channels):
    with tf.variable_scope('up_convolution'):
        x = deconvolution_3d(layer_input, [2, 2, 2, in_channels // 2, in_channels], output_shape, [1, 2, 2, 2, 1])
        return prelu(x)

class V_Net(object):
    def __init__(self, is_training):
        self.is_training = is_training

    def inference_op(self, _input):
        input_channels = 1
        output_channels = 1
        n_channels = 16

        with tf.variable_scope('contracting_path'):

            # if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input channel
            if input_channels == 1:
                c0 = tf.tile(_input, [1, 1, 1, 1, n_channels])
            else:
                with tf.variable_scope('level_0'):
                    c0 = prelu(convolution_3d(_input, [5, 5, 5, input_channels, n_channels], [1, 1, 1, 1, 1]))

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
        with tf.variable_scope('prediction'):
            #softmax_prob = tf.nn.softmax(logits=logits, name='softmax_prob')
            #predicted_label = tf.argmax(input=softmax_prob, axis=4, name='predicted_label')
            predicted_label = tf.nn.sigmoid(logits, name='predicted_label')
        print(logits,predicted_label)
        return logits,predicted_label

    def loss_op(self, logits, labels, dst_weight=None):

        with tf.name_scope("weighted_cross_entropy"):
            if cfg.use_dice_loss == True:
                self.total_loss = tensorlayer_dice_loss(logits[0], labels)
            elif cfg.use_dice_loss == False:
                self.total_loss = entropy_loss_function(logits[0], labels, dst_weight)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.squeeze(tf.cast(logits[1] + 0.5, tf.int32),axis=-1), tf.cast(labels,dtype=tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar("total_loss", self.total_loss)

        return self.total_loss, accuracy