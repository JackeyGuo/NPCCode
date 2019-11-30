import tensorflow as tf
import config.config as cfg
from core.layers import _weight_variable_msra, _batch_norm, conv3d, _create_conv_layer
from core.loss_func import *

class Unet():
    def __init__(self,is_training):
        #super().__init__(is_training)
        #self.cfg = config_holder.get_instance()
        self.now_deep = 0
        self.layer_stack = []
        self.is_training = 'is_training'

    def _create_maxpooling(self,_input):
        if self.now_deep >= len(self.layer_stack):
            self.layer_stack.append([])
        self.layer_stack[self.now_deep].append(_input)
        self.now_deep += 1
        return tf.nn.max_pool3d(_input,[1,2,2,2,1],[1,2,2,2,1],"SAME")

    def _create_up_conv(self,_input,kernel_size=3):
        """
                up conv,with concat
                :param _input:
                :param kernel_size:
                :return:
                """
        out_shape = self.layer_stack[self.now_deep - 1][-1].get_shape()
        shape = [int(out_shape[0]), int(out_shape[1]), int(out_shape[2]), int(out_shape[3]), int(out_shape[4])]

        in_channel = int(_input.get_shape()[-1])
        kernel = _weight_variable_msra([kernel_size, kernel_size, kernel_size, shape[-1], in_channel], "kernel")
        output = tf.nn.conv3d_transpose(_input, kernel, out_shape, [1, 2, 2, 2, 1])
        output = _batch_norm(output)
        output = tf.nn.relu(output)
        output = tf.concat([self.layer_stack[self.now_deep - 1][-1], output],-1)
        self.now_deep -= 1
        return output


    def inference_op(self, _input):
        img = _input
        img_shape = img.get_shape()
        # output = self._create_conv_layer(_input, 5, 2)  # change to 1
        output = _batch_norm(_input)
        output = conv3d(output, kernel_size=5, stride=2, output_feature=16, padding="SAME")
        print(output)
        with tf.variable_scope("conv1"):
            output = _create_conv_layer(output,output_feature=32)
            #print(output)
        with tf.variable_scope("conv2"):
            output = _create_conv_layer(output,output_feature=32)
            #print(output)
        with tf.variable_scope("pool1"):
            output = self._create_maxpooling(output)
            #print(output)
        with tf.variable_scope("conv3"):
            output = _create_conv_layer(output,output_feature=64)
            #print(output)
        with tf.variable_scope("conv4"):
            output = _create_conv_layer(output,output_feature=64)
        with tf.variable_scope("pool2"):
            output = self._create_maxpooling(output)
        with tf.variable_scope("conv5"):
            output = _create_conv_layer(output,output_feature=128)
        with tf.variable_scope("conv6"):
            output = _create_conv_layer(output,output_feature=128)
        with tf.variable_scope("up_conv1"):
            output = self._create_up_conv(output)
        with tf.variable_scope("conv7"):
            output = _create_conv_layer(output,output_feature=64)
        with tf.variable_scope("conv8"):
            output = _create_conv_layer(output,output_feature=64)
        with tf.variable_scope("up_conv2"):
            output = self._create_up_conv(output)
        with tf.variable_scope("conv9"):
            output = _create_conv_layer(output,output_feature=32)
        with tf.variable_scope("conv10"):
            output = _create_conv_layer(output,output_feature=32)
        print(output)
        in_channel = int(output.get_shape()[-1])
        kernel = _weight_variable_msra([3, 3, 3, 8, in_channel], "upconv_kernel")
        output_shape = [int(img_shape[0]), int(img_shape[1]), int(img_shape[2]), int(img_shape[3]), 8]
        output = tf.nn.conv3d_transpose(output, kernel, output_shape, [1, 2, 2, 2, 1])
        output = tf.concat([img, output], -1)
        with tf.variable_scope("conv_bf1"):
            _output = _create_conv_layer(output)
            output = tf.concat([_output, output], -1)
        # with tf.variable_scope("conv_bf2"):
        #     _output = self._create_conv_layer(output)
        #     output = tf.concat([_output, output], -1)
        with tf.variable_scope("fc_layer"):
            output = _create_conv_layer(output, 1, output_feature=2)

        softmax_prob = tf.nn.softmax(logits=output, name='softmax_prob')
        predicted_label = tf.argmax(input=softmax_prob, axis=4, name='predicted_label')

        return output,predicted_label

    def loss_op(self, logits, labels, dst_weight=None):

        #loss = entropy_loss_function(logits, labels, dst_weight)
        loss = softmax_loss_function(logits[0], labels, dst_weight)
        '''with tf.name_scope("cross_entropy"):
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=tf.cast(labels,dtype=tf.int32)))'''

        tf.summary.scalar("total_loss", loss)

        return loss

