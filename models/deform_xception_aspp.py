import numpy as np
import tensorflow as tf

import config.config as cfg
from core.layers import _conv3d, _active_layer, deconv3d, conv_bn_prelu, deconv_bn_prelu, atrous_bn_prelu, deform_conv3d, BilinearUpsample3d
from core.loss_func import *

class deform_xception_aspp(object):
    def __init__(self, is_training):

        self.layer_stack = []
        self.cont_block_num = 3
        self.expand_block_num = 3
        #self.loss_coefficient = 1e4
        self.is_training = is_training
        self.atrou_num = 3

        self.output_channels = {'1': [16, 16, 16], '2': [32, 32, 32], '3': [64, 64, 64], '4': [32, 32], '5': [32, 16],'6': [16, 8]}
        self.output_labels = 2
        #self.output_channels = {'1': [8, 8, 8], '2': [16, 16, 16], '3': [32, 32, 32], '4': [16, 16], '5': [16, 8],
                           # '6': [8, 4]}
    def feature_pyramid(self,_input,block_num):
        atrous_layer = []
        out_feature = int(_input.get_shape()[-1])

        output_1x1 = _conv3d(_input, kernel_size=1, stride=1, output_feature=out_feature, use_bias=True, name='pyramid_conv_1')

        for i in range(1, self.atrou_num + 1):
            dilate_rate = int(np.power(2,4-block_num)*i)
            #print(dilate_rate)
            # dilate rate : 24 16 8 / 12 8 4 / 6 4 2
            output = atrous_bn_prelu(_input, kernel_size=3, stride=1, output_channels=out_feature/2,
                             dilation_rate=dilate_rate, is_training=self.is_training,name='atrous_conv%d' % i)

            atrous_layer.append(output)

        output = tf.concat([output_1x1, atrous_layer[0], atrous_layer[1], atrous_layer[2]], axis=-1)
        print('atrous conv shape:', output)
        output = _conv3d(output, kernel_size=1, stride=1, output_feature=out_feature,use_bias=True, name='pyramid_conv_1x1')

        return output
    def cascade_feature(self,_input,block_num):
        atrous_layer = []
        out_feature = int(_input.get_shape()[-1])
        output = _input
        #output_1x1 = _conv3d(_input, kernel_size=1, stride=1, output_feature=out_feature, use_bias=True, name='pyramid_conv_1')

        for i in range(1, self.atrou_num + 1):
            dilate_rate = int(np.power(2,4-block_num)*i)
            #print(dilate_rate)
            # dilate rate : 24 16 8 / 12 8 4 / 6 4 2
            output = atrous_bn_prelu(output, kernel_size=3, stride=1, output_channels=out_feature,
                             dilation_rate=dilate_rate, is_training=self.is_training,name='atrous_conv%d' % i)

            atrous_layer.append(output)

        output = tf.concat([atrous_layer[0], atrous_layer[1], atrous_layer[2]], axis=-1)
        print('atrous conv shape:', output)
        output = _conv3d(output, kernel_size=1, stride=1, output_feature=out_feature,use_bias=True, name='pyramid_conv_1x1')

        return output
    def _contracting_block(self, block_num, _input):
        # xception contracte blcok
        # 52*100*80*16 / 26*50*40*32 / 13*25*20*64
        output = conv_bn_prelu(_input, kernel_size=3, stride=2, output_channels=self.output_channels[str(block_num)][0],
                         name='conv_1', is_training=self.is_training)
        # 52*100*80*8 / 26*50*40*32 / 13*25*20*64
        # 52*100*80*16 / 26*50*40*32 / 13*25*20*64
        output = conv_bn_prelu(output, kernel_size=3, stride=1, output_channels=self.output_channels[str(block_num)][1],
                               name='conv_2', is_training=self.is_training)
        output = conv_bn_prelu(output, kernel_size=3, stride=1, output_channels=self.output_channels[str(block_num)][2],
                               name='conv_3', is_training=self.is_training)
        #output = self.feature_pyramid(output)
        output = self.cascade_feature(output, block_num)
        # do conv 1 x 1 x 1 before sum
        #input = _conv3d(_input, kernel_size=1, stride=2, output_feature=self.output_channels[str(block_num)][2],
                        #use_bias=True, name='conv_s2')
        input = conv_bn_prelu(_input, kernel_size=3, stride=2, output_channels=self.output_channels[str(block_num)][2],
                                                    name='conv_s2', is_training=self.is_training)
        output = tf.add(output, input, name='elemwise_sum')

        #output = self.cascade_feature(output,block_num)
        #layer.append(output)
        return output

    def _expanding_block(self, block_num, _input, layer, option='concat'):

        # 13*25*20*32 / 26*50*40*32 / 52*100*80*16
        #output = conv_bn_prelu(_input, kernel_size=1, stride=1, output_channels=self.output_channels[str(block_num)][0],
                              # name='conv_1', is_training=self.is_training)
        output = _conv3d(_input, kernel_size=1, stride=1, output_feature=self.output_channels[str(block_num)][0],
                        use_bias=True, name='conv_1')
        output = deconv_bn_prelu(output, output_channels=self.output_channels[str(block_num)][1],
                                 is_training=self.is_training, name='deconv')
        # 26*50*40*32 / 52*100*80*16 / 104*200*160*8
        # 26*50*40*64 / 52*100*80*32 / 104*200*160*16
        output = tf.concat(values=(output, layer), axis=4, name='concat')
        # 26*50*40*64 / 52*100*80*32 / 104*200*160*16
        output = conv_bn_prelu(output, kernel_size=3, stride=1, output_channels=int(output.get_shape()[-1]),
                               name='conv_2', is_training=self.is_training)
        output = conv_bn_prelu(output, kernel_size=3, stride=1, output_channels=int(output.get_shape()[-1]),
                               name='conv_3', is_training=self.is_training)
        print(output)
        return output

    def inference_op(self, _input):

        conv_layer = []
        dconv_layer = []

        # padding output
        output = tf.pad(_input, np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0]]), name='pad_1')
        output = conv_bn_prelu(output, output_channels=8, kernel_size=3, stride=1,
                               is_training=self.is_training, name='conv_1') # 104x200x160 8
        conv_layer.append(output)

        for block_num in range(1, self.cont_block_num + 1):
            with tf.variable_scope('contract_block_%d' % block_num):
                output = self._contracting_block(block_num,output)
                conv_layer.append(output)

        for block_num in range(4, self.expand_block_num + 4):
            with tf.variable_scope('expand_block_%d' % block_num):
                output = self._expanding_block(block_num, output, conv_layer[2 - block_num])
                dconv_layer.append(output)

        '''auxiliary prediction'''
        # forth level
        auxiliary3_prob_4x = _conv3d(inputs=dconv_layer[0], output_feature=self.output_labels, kernel_size=1,
                                     stride=1, use_bias=True, name='auxiliary3_prob_4x')
        auxiliary3_prob_2x = deconv3d(inputs=auxiliary3_prob_4x, output_channels=self.output_labels,
                                      name='auxiliary3_prob_2x')
        auxiliary3_prob_1x = deconv3d(inputs=auxiliary3_prob_2x, output_channels=self.output_labels,
                                      name='auxiliary3_prob_1x')
        # third level
        auxiliary2_prob_2x = _conv3d(inputs=dconv_layer[1], output_feature=self.output_labels, kernel_size=1,
                                     stride=1, use_bias=True, name='auxiliary2_prob_2x')
        auxiliary2_prob_1x = deconv3d(inputs=auxiliary2_prob_2x, output_channels=self.output_labels,
                                      name='auxiliary2_prob_2x')
        # second level
        auxiliary1_prob_1x = _conv3d(inputs=dconv_layer[2], output_feature=self.output_labels, kernel_size=1,
                                     stride=1, use_bias=True, name='auxiliary1_prob_1x')

        with tf.variable_scope('last_stage'):
            # out_feature = int(output.get_shape()[-1]) / 2
            #print(dconv_layer[0],dconv_layer[1],dconv_layer[2])
            output1 = _conv3d(dconv_layer[0], kernel_size=1, stride=1, output_feature=5, use_bias=True,
                              name='block1_conv1x1')

            #output1 = deconv3d(output1, output_channels=int(output1.get_shape()[-1]), name='block1_deconv')

            output1 = BilinearUpsample3d(output1,up_factor=2)

            output2 = _conv3d(dconv_layer[1], kernel_size=1, stride=1, output_feature=5, use_bias=True,
                               name='block2_conv1x1')

            output2 = tf.add(output1, output2)

            #output2 = deconv3d(output2, output_channels=int(output2.get_shape()[-1]), name='block2_deconv')
            output2 = BilinearUpsample3d(output2, up_factor=2)

            output3 = _conv3d(dconv_layer[2], kernel_size=1, stride=1, output_feature=5, use_bias=True,
                               name='block3_conv1x1')

            output3 = tf.add(output2, output3)

            output = _conv3d(output3, kernel_size=1, stride=1, output_feature=self.output_labels, use_bias=True, name='fc_layer')
            output = output[:, :103, :198, :]
        with tf.device(device_name_or_function='/cpu:0'):
            softmax_prob = tf.nn.softmax(logits=output, name='softmax_prob')
            predicted_label = tf.argmax(input=softmax_prob, axis=4, name='predicted_label')
        #print(output,predicted_label,auxiliary3_prob_1x,auxiliary1_prob_1x,auxiliary2_prob_1x)
        return output, predicted_label, auxiliary1_prob_1x, auxiliary2_prob_1x, auxiliary3_prob_1x

    def loss_op(self, outputs, labels, dst_weight=None):
        logits = outputs[0][:, :103, : 198, :]
        pred_labels = outputs[1][:, :103, :198, :]
        auxiliary1_prob_1x = outputs[2][:, :103, : 198, :]
        auxiliary2_prob_1x = outputs[3][:, :103, : 198, :]
        auxiliary3_prob_1x = outputs[4][:, :103, : 198, :]

        # dice loss
        ''''self.main_dice_loss = tensorlayer_dice_loss(logits, labels)
        self.auxiliary1_dice_loss = tensorlayer_dice_loss(auxiliary1_prob_1x, labels)
        self.auxiliary2_dice_loss = tensorlayer_dice_loss(auxiliary2_prob_1x, labels)
        self.auxiliary3_dice_loss = tensorlayer_dice_loss(auxiliary3_prob_1x, labels)
        self.total_dice_loss = \
            self.main_dice_loss + \
            self.auxiliary1_dice_loss * 0.8 + \
            self.auxiliary2_dice_loss * 0.4 + \
            self.auxiliary3_dice_loss * 0.2'''

        # class-weighted cross-entropy loss
        '''self.main_weight_loss = entropy_loss_function(logits, labels, dst_weight)

        self.auxiliary1_weight_loss = entropy_loss_function(auxiliary1_prob_1x, labels, dst_weight)
        self.auxiliary2_weight_loss = entropy_loss_function(auxiliary2_prob_1x, labels, dst_weight)
        self.auxiliary3_weight_loss = entropy_loss_function(auxiliary3_prob_1x, labels, dst_weight)
        self.total_weight_loss = \
            self.main_weight_loss + \
            self.auxiliary1_weight_loss * 0.9 + \
            self.auxiliary2_weight_loss * 0.6 + \
            self.auxiliary3_weight_loss * 0.3'''

        # class-weighted focal loss
        '''self.main_focal_loss = focal_loss(logits, labels)

        self.auxiliary1_focal_loss = focal_loss(auxiliary1_prob_1x, labels)
        self.auxiliary2_focal_loss = focal_loss(auxiliary2_prob_1x, labels)
        self.auxiliary3_focal_loss = focal_loss(auxiliary3_prob_1x, labels)
        self.total_focal_loss = \
            self.main_focal_loss + \
            self.auxiliary1_focal_loss * 0.9 + \
            self.auxiliary2_focal_loss * 0.6 + \
            self.auxiliary3_focal_loss * 0.3'''

        self.main_jacc_loss = jacc_loss(logits, labels)
        self.auxiliary1_jacc_loss = jacc_loss(auxiliary1_prob_1x, labels)
        self.auxiliary2_jacc_loss = jacc_loss(auxiliary2_prob_1x, labels)
        self.auxiliary3_jacc_loss = jacc_loss(auxiliary3_prob_1x, labels)
        self.total_jacc_loss = \
            self.main_jacc_loss + \
            self.auxiliary1_jacc_loss * 0.8 + \
            self.auxiliary2_jacc_loss * 0.4 + \
            self.auxiliary3_jacc_loss * 0.2
        #self.main_jacc_loss = softmax_loss_function(logits, labels)

        if cfg.use_focal_loss == True:
            self.total_loss = self.total_dice_loss + self.total_focal_loss

        elif cfg.only_use_dice == True:
            self.total_loss = self.total_dice_loss

        elif cfg.only_jacc_loss == True:
            self.total_loss = self.total_jacc_loss

        elif cfg.jacc_entropy_loss == True:
            self.total_loss = self.total_jacc_loss + self.total_weight_loss
        else:
            self.total_loss = self.total_dice_loss + self.total_weight_loss

        with tf.name_scope("accuracy"):
            if cfg.predict_op == 'sigmoid':
                correct_pred = tf.equal(tf.squeeze(tf.cast(pred_labels + 0.5, tf.int32),axis=-1), tf.cast(labels,dtype=tf.int32))
            else:
                #correct_pred = tf.equal(pred_labels,tf.cast(labels, dtype=tf.int64))
                correct_pred = tf.reduce_sum(pred_labels) / tf.reduce_sum(tf.cast(labels, dtype=tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('total loss', self.total_loss )
        #tf.summary.scalar("total_jacc_loss", self.total_jacc_loss)
        #tf.summary.scalar("main_dice_loss", self.main_dice_loss)
        #tf.summary.scalar("total_dice_loss", self.total_dice_loss)
        #tf.summary.scalar("total_weight_loss", self.total_weight_loss)
        #tf.summary.scalar("total_jacc_loss", self.total_jacc_loss)
        #tf.summary.scalar("total_focal_loss", self.total_focal_loss)

        return self.total_loss,accuracy
