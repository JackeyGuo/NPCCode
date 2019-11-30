import numpy as np
import tensorflow as tf

import config.config as cfg
from core.layers import _conv3d, _active_layer, deconv3d, conv_bn_prelu
from core.loss_func import *

class cnn(object):
    def __init__(self,is_training):

        self.layer_stack = []
        self.cont_block_num = 3
        self.expand_block_num = 3
        self.loss_coefficient = 1e4
        self.is_training = is_training
        self.atrou_num = 3

    def _contracting_block(self, _input, option='concat'):
        out_feature = _input.get_shape()[-1]
        #print('out_feature', out_feature)

        layer = []
        atrous_layer = []

        output = _conv3d(_input, kernel_size=3, stride=2, output_feature=out_feature, name='conv_1')
        #print(output)
        layer.append(output)
        output = _active_layer(output, name='active_layer_1', is_training=self.is_training)
        output = _conv3d(output, kernel_size=3, stride=1, output_feature=out_feature, name='conv_2')
        #print(output)
        if option == 'sum':
            output = tf.add(output, layer[-1], name='elemwise_sum')
            #print(output)
        else:
            output = tf.concat(values=(output, layer[-1]), axis=4, name='concat')

        output = _active_layer(output, name='active_layer_2', is_training=self.is_training)

        '''for i in range(self.atrou_num):

            output = self._create_conv_layer(_input, atrous=2 ** i)
            output = _conv3d(output, kernel_size=3, stride=1, output_feature=out_feature, dilation_rate=2 ** i,
                             name='atrous_conv%d'%i)
            atrous_layer.append(output)'''

        #output = tf.concat(atrous_layer,axis=-1)
        print(output)

        layer.append(output)
        return output
    def _expanding_block(self, _input, layer, option='concat'):

        out_feature = int(_input.get_shape()[-1]) / 2

        output = _conv3d(_input, kernel_size=1, stride=1, output_feature=out_feature, name='conv_1')
        #print(output)
        output = _active_layer(output, name='active_layer_1', is_training=self.is_training)
        #output = _create_dconv_layer(output)
        output = deconv3d(output, output_channels=int(output.get_shape()[-1]), name='deconv')
        #$print(output)
        output = _active_layer(output, name='active_layer_2', is_training=self.is_training)

        if option == 'sum':
            output = tf.add(output,layer, name='elemwise_sum')
            #print(output)
        else:
            output = tf.concat(values=(output, layer), axis=4, name='concat')

        output = _conv3d(output, kernel_size=3, stride=1,
                         output_feature=int(output.get_shape()[-1]), name='conv_2')
        #print(output)
        output = _active_layer(output, name='active_layer_3', is_training=self.is_training)
        return output
    def inference_op(self,_input):

        conv_layer = []
        dconv_layer = []

        # padding output
        output = tf.pad(_input, np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0]]), name='pad_1')
        #print(output)
        output = conv_bn_prelu(output, output_channels=8, kernel_size=3, stride=1,
                               is_training=self.is_training, name='conv_1')
        conv_layer.append(output)

        for i in range(self.cont_block_num):
            with tf.variable_scope('contract_block_%d'%i):
                output = self._contracting_block(output)
                #print(output)
                conv_layer.append(output)

        for i in range(self.expand_block_num):
            with tf.variable_scope('expand_block_%d'%i):
                output = self._expanding_block(output,conv_layer[2-i])
                dconv_layer.append(output)

        '''auxiliary prediction'''
        # forth level
        auxiliary3_prob_4x = _conv3d(inputs=dconv_layer[0], output_feature=1, kernel_size=1,
                                    stride=1, use_bias=True, name='auxiliary3_prob_4x')
        auxiliary3_prob_2x = deconv3d(inputs=auxiliary3_prob_4x, output_channels=1,
                                      name='auxiliary3_prob_2x')
        auxiliary3_prob_1x = deconv3d(inputs=auxiliary3_prob_2x, output_channels=1,
                                      name='auxiliary3_prob_1x')
        # third level
        auxiliary2_prob_2x = _conv3d(inputs=dconv_layer[1], output_feature=1, kernel_size=1,
                                    stride=1, use_bias=True, name='auxiliary2_prob_2x')
        auxiliary2_prob_1x = deconv3d(inputs=auxiliary2_prob_2x, output_channels=1,
                                      name='auxiliary2_prob_2x')
        # second level
        auxiliary1_prob_1x = _conv3d(inputs=dconv_layer[2], output_feature=1, kernel_size=1,
                                    stride=1, use_bias=True, name='auxiliary1_prob_1x')

        #print(auxiliary3_prob_1x,'\n',auxiliary2_prob_1x,'\n',auxiliary1_prob_1x)
        with tf.variable_scope('last_stage'):
            # out_feature = int(output.get_shape()[-1]) / 2
            _output = _conv3d(dconv_layer[0], kernel_size=1, stride=1, output_feature=32, use_bias=True, name='block1_conv1x1')

            _output = deconv3d(_output, output_channels=int(_output.get_shape()[-1]), name='block1_deconv')
            #print('block1_deconv1', _output)

            # out_feature = int(output.get_shape()[-1]) / 2
            _output2 = _conv3d(dconv_layer[1], kernel_size=1, stride=1, output_feature=32, use_bias=True, name='block2_conv1x1')

            _output = tf.add(_output, _output2)
            #print('1', _output)

            _output = deconv3d(_output, output_channels=int(_output.get_shape()[-1]), name='block2_deconv')

            # out_feature = int(output.get_shape()[-1]) / 2
            _output3 = _conv3d(dconv_layer[2], kernel_size=1, stride=1, output_feature=32, use_bias=True, name='block3_conv1x1')

            output = tf.add(_output, _output3)

            output = _conv3d(output, kernel_size=1, stride=1, output_feature=1, use_bias=True, name='fc_layer')

        #logits = tf.nn.sigmoid(output)'''

        return output, auxiliary1_prob_1x, auxiliary2_prob_1x, auxiliary3_prob_1x

    def loss_op(self, outputs, labels):
        logits = outputs[0][:, :103, : 198, :]
        auxiliary1_prob_1x = outputs[1][:, :103, : 198, :]
        auxiliary2_prob_1x = outputs[2][:, :103, : 198, :]
        auxiliary3_prob_1x = outputs[3][:, :103, : 198, :]

        # dice loss
        '''self.main_dice_loss = tensorlayer_dice_loss(logits, labels)
        self.auxiliary1_dice_loss = tensorlayer_dice_loss(auxiliary1_prob_1x, labels)
        self.auxiliary2_dice_loss = tensorlayer_dice_loss(auxiliary2_prob_1x, labels)
        self.auxiliary3_dice_loss = tensorlayer_dice_loss(auxiliary3_prob_1x, labels)
        self.total_dice_loss = \
            self.main_dice_loss + \
            self.auxiliary1_dice_loss * 0.8 + \
            self.auxiliary2_dice_loss * 0.4 + \
            self.auxiliary3_dice_loss * 0.2
        # class-weighted cross-entropy loss
        self.main_weight_loss = entropy_loss_function(logits,labels)

        self.auxiliary1_weight_loss = entropy_loss_function(auxiliary1_prob_1x, labels)
        self.auxiliary2_weight_loss = entropy_loss_function(auxiliary2_prob_1x, labels)
        self.auxiliary3_weight_loss = entropy_loss_function(auxiliary3_prob_1x, labels)
        self.total_weight_loss = \
            self.main_weight_loss + \
            self.auxiliary1_weight_loss * 0.9 + \
            self.auxiliary2_weight_loss * 0.6 + \
            self.auxiliary3_weight_loss * 0.3'''

        #self.dice_loss = tensorlayer_dice_loss(logits=logits,labels=labels)
        #self.total_loss = self.total_dice_loss + self.total_weight_loss

        self.total_loss = self.main_weight_loss + self.main_dice_loss

        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("total_dice_loss", self.total_dice_loss)
        tf.summary.scalar("total_weight_loss", self.total_weight_loss)
        tf.summary.scalar("main_weight_loss", self.main_weight_loss)
        tf.summary.scalar("main_dice_loss", self.main_dice_loss)

        return self.total_loss
'''# TODO:remove receptive_field! or put it into config file.
        receptive_field = 0
        half_receptive = receptive_field // 2

        data_shape = labels.get_shape()
        transed_shape = [int(data_shape[0]), int(data_shape[1]), int(data_shape[2]), int(data_shape[3])]
        # slice the logits and labels to the non-padding area.
        logits_slice = tf.slice(logits, [0, half_receptive, half_receptive, half_receptive, 0],
                                [1, transed_shape[1] - receptive_field, transed_shape[2] - receptive_field,
                                 transed_shape[3] - receptive_field, 1])

        labels_slice = tf.slice(labels, [0, half_receptive, half_receptive, half_receptive],
                                [1, transed_shape[1] - receptive_field, transed_shape[2] - receptive_field,
                                 transed_shape[3] - receptive_field])

        logits_slice = tf.squeeze(logits_slice,axis=-1)
        #labels_slice = tf.squeeze(labels_slice,axis=-1)
        labels_slice = tf.cast(labels_slice, dtype=tf.float32)

        weight_loss = cfg.weight_loss
        #pw = 1
        if weight_loss == 0:
            pw = (1 - tf.reduce_mean(labels_slice)) / tf.reduce_mean(labels_slice)
            # change it to calc weight.
        else:
            pw = weight_loss

        loss = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(targets=labels_slice, logits=logits_slice, pos_weight=pw))
        tf.summary.scalar("total_loss", loss)'''