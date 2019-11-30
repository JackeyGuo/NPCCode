import numpy as np
import tensorflow as tf
from core.conv_def import conv_bn_relu,deconv_bn_relu,conv3d
from core.loss_func import entropy_loss_function,softmax_loss_function,jacc_loss
import config.config as cfg
from tools.utils import DataProvider

class cnn(object):
    def __init__(self,is_training):
        self.is_training = is_training
        self.loss_coefficient = 1e4
        # predefined
        # single-gpu
        self.gpu_number = len(cfg.gpu.split(','))
        if self.gpu_number > 1:
            self.device = ['/gpu:0', '/gpu:1', '/cpu:0']
        else:
            self.device = ['/gpu:0', '/gpu:0', '/cpu:0']

        self.output_channels = 2

    def inference_op(self,_input):
        # padding output
        _input = tf.pad(_input, np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0]]), name='pad_1')

        concat_dimension = 4  # channels_last
        # padding output
        with tf.device(device_name_or_function=self.device[0]):
            # first level
            encoder1_1 = conv_bn_relu(inputs=_input, output_channels=16, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='encoder1_1')
            encoder1_2 = conv_bn_relu(inputs=encoder1_1, output_channels=32, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='encoder1_2')
            pool1 = tf.layers.max_pooling3d(
                inputs=encoder1_2,
                pool_size=2,                    # pool_depth, pool_height, pool_width
                strides=2,
                padding='valid',                # No padding, default
                data_format='channels_last',    # default
                name='pool1'
            )
            # second level
            encoder2_1 = conv_bn_relu(inputs=pool1, output_channels=32, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='encoder2_1')
            encoder2_2 = conv_bn_relu(inputs=encoder2_1, output_channels=64, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='encoder2_2')
            pool2 = tf.layers.max_pooling3d(inputs=encoder2_2, pool_size=2, strides=2, name='pool2')
            # third level
            encoder3_1 = conv_bn_relu(inputs=pool2, output_channels=64, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='encoder3_1')
            encoder3_2 = conv_bn_relu(inputs=encoder3_1, output_channels=128, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='encoder3_2')
            pool3 = tf.layers.max_pooling3d(inputs=encoder3_2, pool_size=2, strides=2, name='pool3')
            # forth level
            encoder4_1 = conv_bn_relu(inputs=pool3, output_channels=128, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='encoder4_1')
            encoder4_2 = conv_bn_relu(inputs=encoder4_1, output_channels=256, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='encoder4_2')
            bottom = encoder4_2

        # up-sampling path
        # device: gpu1
        with tf.device(device_name_or_function=self.device[1]):
            # third level
            deconv3 = deconv_bn_relu(inputs=bottom, output_channels=256, is_training=self.is_training,
                                     name='deconv3')
            concat_3 = tf.concat([deconv3, encoder3_2], axis=concat_dimension, name='concat_3')
            decoder3_1 = conv_bn_relu(inputs=concat_3, output_channels=128, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='decoder3_1')
            decoder3_2 = conv_bn_relu(inputs=decoder3_1, output_channels=128, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='decoder3_2')
            # second level
            deconv2 = deconv_bn_relu(inputs=decoder3_2, output_channels=128, is_training=self.is_training,
                                     name='deconv2')
            concat_2 = tf.concat([deconv2, encoder2_2], axis=concat_dimension, name='concat_2')
            decoder2_1 = conv_bn_relu(inputs=concat_2, output_channels=64, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='decoder2_1')
            decoder2_2 = conv_bn_relu(inputs=decoder2_1, output_channels=64, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='decoder2_2')
            # first level
            deconv1 = deconv_bn_relu(inputs=decoder2_2, output_channels=64, is_training=self.is_training,
                                     name='deconv1')
            concat_1 = tf.concat([deconv1, encoder1_2], axis=concat_dimension, name='concat_1')
            decoder1_1 = conv_bn_relu(inputs=concat_1, output_channels=32, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='decoder1_1')
            decoder1_2 = conv_bn_relu(inputs=decoder1_1, output_channels=32, kernel_size=3, stride=1,
                                      is_training=self.is_training, name='decoder1_2')
            feature = decoder1_2
            # predicted probability
            predicted_prob = conv3d(inputs=feature, output_channels=self.output_channels, kernel_size=1,
                                    stride=1, use_bias=True, name='predicted_prob')
            #print(predicted_prob)
            predicted_prob = predicted_prob[:, :103, :198, :]
            #print(predicted_prob)

            ''''auxiliary prediction'''
            # forth level
            '''auxiliary3_prob_8x = conv3d(inputs=encoder4_2, output_channels=self.output_channels, kernel_size=1,
                                        stride=1, use_bias=True, name='auxiliary3_prob_8x')
            auxiliary3_prob_4x = deconv3d(inputs=auxiliary3_prob_8x, output_channels=self.output_channels,
                                          name='auxiliary3_prob_4x')
            auxiliary3_prob_2x = deconv3d(inputs=auxiliary3_prob_4x, output_channels=self.output_channels,
                                          name='auxiliary3_prob_2x')
            auxiliary3_prob_1x = deconv3d(inputs=auxiliary3_prob_2x, output_channels=self.output_channels,
                                          name='auxiliary3_prob_1x')
            # third level
            auxiliary2_prob_4x = conv3d(inputs=decoder3_2, output_channels=self.output_channels, kernel_size=1,
                                        stride=1, use_bias=True, name='auxiliary2_prob_4x')
            auxiliary2_prob_2x = deconv3d(inputs=auxiliary2_prob_4x, output_channels=self.output_channels,
                                          name='auxiliary2_prob_2x')
            auxiliary2_prob_1x = deconv3d(inputs=auxiliary2_prob_2x, output_channels=self.output_channels,
                                          name='auxiliary2_prob_1x')
            # second level
            auxiliary1_prob_2x = conv3d(inputs=decoder2_2, output_channels=self.output_channels, kernel_size=1,
                                        stride=1, use_bias=True, name='auxiliary1_prob_2x')
            auxiliary1_prob_1x = deconv3d(inputs=auxiliary1_prob_2x, output_channels=self.output_channels,
                                          name='auxiliary1_prob_1x')
        #print(predicted_prob)
        with tf.device(device_name_or_function=self.device[2]):
            softmax_prob = tf.nn.softmax(logits=predicted_prob, name='softmax_prob')
            #sigmoid = tf.nn.sigmoid(predicted_prob)
            #predicted_label = softmax_prob
            predicted_label = tf.argmax(input=softmax_prob, axis=4, name='predicted_label')'''
        with tf.variable_scope('prediction'):
            softmax_prob = tf.nn.softmax(logits=predicted_prob, name='softmax_prob')
            predicted_label = tf.argmax(input=softmax_prob, axis=4, name='predicted_label')
            #predicted_label = tf.nn.sigmoid(predicted_prob, name='predicted_label')
        return predicted_prob,predicted_label
        #return predicted_prob, predicted_label, auxiliary1_prob_1x, auxiliary2_prob_1x, auxiliary3_prob_1x

    def build_model(self, outputs, labels, dst_weight=None):
        # input data and labels
        '''self.input_image = input_image
        self.input_ground_truth = true_label

        #self.input_ground_truth = tf.pad(self.input_ground_truth, np.array([[0, 0], [1, 0], [1, 1], [0, 0]]), name='pad_2')

        # probability
        self.predicted_prob, self.predicted_label, self.auxiliary1_prob_1x, \
            self.auxiliary2_prob_1x, self.auxiliary3_prob_1x = self.unet_model(self.input_image)'''

        self.auxiliary1_prob_1x = self.auxiliary1_prob_1x[:, :103, :198, :]
        self.auxiliary2_prob_1x = self.auxiliary2_prob_1x[:, :103, :198, :]
        self.auxiliary3_prob_1x = self.auxiliary3_prob_1x[:, :103, :198, :]

        # dice loss
        '''self.main_dice_loss = dice_loss_function(self.predicted_prob, self.input_ground_truth)
        self.auxiliary1_dice_loss = dice_loss_function(self.auxiliary1_prob_1x, self.input_ground_truth)
        self.auxiliary2_dice_loss = dice_loss_function(self.auxiliary2_prob_1x, self.input_ground_truth)
        self.auxiliary3_dice_loss = dice_loss_function(self.auxiliary3_prob_1x, self.input_ground_truth)
        self.total_dice_loss = \
            self.main_dice_loss + \
            self.auxiliary1_dice_loss * 0.8 + \
            self.auxiliary2_dice_loss * 0.4 + \
            self.auxiliary3_dice_loss * 0.2
        # class-weighted cross-entropy loss
        self.main_weight_loss = softmax_loss_function(self.predicted_prob, self.input_ground_truth)

        self.auxiliary1_weight_loss = softmax_loss_function(self.auxiliary1_prob_1x, self.input_ground_truth)
        self.auxiliary2_weight_loss = softmax_loss_function(self.auxiliary2_prob_1x, self.input_ground_truth)
        self.auxiliary3_weight_loss = softmax_loss_function(self.auxiliary3_prob_1x, self.input_ground_truth)
        self.total_weight_loss = \
            self.main_weight_loss +\
            self.auxiliary1_weight_loss * 0.9 + \
            self.auxiliary2_weight_loss * 0.6 + \
            self.auxiliary3_weight_loss * 0.3

        self.dice_coefficient = dice_coefficient(self.predicted_prob, self.input_ground_truth)
        # TODO: adjust the weights
        self.total_loss = self.total_dice_loss * self.loss_coefficient + self.total_weight_loss'''

        # trainable variables
        #self.trainable_variables = tf.trainable_variables()

        self.entropy_loss = entropy_loss_function(self.predicted_prob, self.input_ground_truth)

        tf.summary.scalar("total_loss", self.entropy_loss)

        #tf.summary.scalar("accuracy", self.acc)
        # TODO: how to extract layers for fine-tuning? why?

        '''How to list all of them'''
        '''fine_tuning_layer = [
                'encoder1_1/encoder1_1_conv/kernel:0',
                'encoder1_2/encoder1_2_conv/kernel:0',
                'encoder2_1/encoder2_1_conv/kernel:0',
                'encoder2_2/encoder2_2_conv/kernel:0',
                'encoder3_1/encoder3_1_conv/kernel:0',
                'encoder3_2/encoder3_2_conv/kernel:0',
                'encoder4_1/encoder4_1_conv/kernel:0',
                'encoder4_2/encoder4_2_conv/kernel:0',
        ]

        # TODO: what does this part mean
        self.fine_tuning_variables = []
        for variable in self.trainable_variables:
            # print('\'%s\',' % variable.name)
            for index, kernel_name in enumerate(fine_tuning_layer):
                if kernel_name in variable.name:
                    self.fine_tuning_variables.append(variable)
                    break  # not necessary to continue

        self.saver = tf.train.Saver()
        self.saver_fine_tuning = tf.train.Saver(self.fine_tuning_variables)
        # The Saver class adds ops to save and restore variables to and from checkpoints.
        # It also provides convenience methods to run these ops.'''

    def loss_op(self, logits, labels, dst_weight=None):

        with tf.name_scope("weighted_cross_entropy"):
            if cfg.use_jacc_loss == False:
                self.total_loss = softmax_loss_function(logits[0], labels, dst_weight)
            else:
                self.total_loss = jacc_loss(logits[0], labels)

        with tf.name_scope("accuracy"):
            if cfg.predict_op == 'sigmoid':
                correct_pred = tf.equal(tf.squeeze(tf.cast(logits[1] + 0.5, tf.int32),axis=-1), tf.cast(labels,dtype=tf.int32))
            else:
                #correct_pred = tf.equal(logits[1],tf.cast(labels, dtype=tf.int64))
                correct_pred = tf.reduce_sum(logits[1]) / tf.reduce_sum(tf.cast(labels, dtype=tf.int64))
                #dice = tensorlayer_dice_loss(logits[1],labels)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        #tf.summary.scalar('accuracy', dice)
        tf.summary.scalar("total_loss", self.total_loss)

        return self.total_loss, accuracy