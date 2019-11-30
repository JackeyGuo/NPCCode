import tensorflow as tf
from config import config as cfg
from core.loss_func import entropy_loss_function,softmax_loss_function,jacc_loss

class DenseDilatedASPP(object):
    def __init__(self,is_training):
        self.block_size = cfg.block_size
        self.block_count = cfg.block_count
        self.use_bc = cfg.use_bc
        self.block_output = []
        self.is_training = is_training
    @staticmethod
    def _weight_variable_msra(shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())

    def _batch_norm(self, _input):
        output = tf.contrib.layers.batch_norm(
            _input, 0.99, scale=True,updates_collections=None)
        return output

    def _conv3d(self, _input, kernel_size, stride, output_feature, padding="SAME"):
        in_features = int(_input.get_shape()[-1])
        kernel = self._weight_variable_msra(
            [kernel_size, kernel_size, kernel_size, in_features, output_feature],
            name='kernel')
        strides = [1, stride, stride, stride, 1]
        output = tf.nn.conv3d(_input, kernel, strides, padding)
        return output

    def _conv3d_atrous(self, _input, kernel_size=3, stride=1, atrous=1, output_feature=8, padding="SAME"):
        in_features = int(_input.get_shape()[-1])
        kernel = self._weight_variable_msra(
            [kernel_size, kernel_size, kernel_size, in_features, output_feature],
            name='kernel')
        strides = [stride, stride, stride]
        atrous_size = [atrous, atrous, atrous]
        return tf.nn.convolution(_input, kernel, padding, strides, atrous_size)

    def _create_conv_layer(self, _input, kernel_size=3, stride=1, atrous=1, output_feature=16, padding="SAME"):
        _input = self._batch_norm(_input)
        _input = tf.nn.relu(_input)
        if atrous == 1:
            output = self._conv3d(_input, kernel_size, stride, output_feature, padding)
        else:
            output = self._conv3d_atrous(_input, kernel_size, stride, atrous, output_feature, padding)
        return output

    def _create_bc_layer(self, _input, output_feature):
        return self._create_conv_layer(_input, 1, output_feature=output_feature)

    def _create_pyramid(self, _input, magic_size):
        layers = []
        for i in range(magic_size):
            with tf.variable_scope("pyramid{0}".format(i)):
                output = self._create_conv_layer(_input, atrous=2 ** i)
                layers.append(output)
        output = tf.concat(layers, -1)
        return output

    def _create_block(self, _input, idx):
        """
        :param _input:
        :param idx:int, 1 block 1 idx.
        :return:
        """
        with tf.variable_scope("block{0}".format(idx)):
            layers = []
            for i in range(self.block_size):
                if i == self.block_size - 1 and idx > 0:
                    _input = self._create_pyramid(_input, idx + 1)
                    _input = tf.concat([layers[-1], _input], -1)
                else:
                    with tf.variable_scope("conv{0}".format(i)):
                        _input = self._create_conv_layer(_input)
                        if len(layers) > 0:
                            _input = tf.concat([layers[-1], _input], -1)
                            layers.append(_input)
                        else:
                            layers.append(_input)

        if self.use_bc:
            with tf.variable_scope("bc_layer{0}".format(idx)):
                output = self._create_bc_layer(_input, 8)
        else:
            output = _input
        return output

    def inference_op(self, _input):
        img = _input
        img_shape = img.get_shape()
        # output = self._create_conv_layer(_input, 5, 2)  # change to 1
        output = self._batch_norm(_input)
        output = self._conv3d(output, 5, 2, 16, "SAME")
        self.block_output.append(output)
        for i in range(self.block_count):
            output = self._create_block(output, i)
            output = tf.concat([self.block_output[-1], output], -1)
        #print('output',output)

        in_channel = int(output.get_shape()[-1])
        kernel = self._weight_variable_msra([3, 3, 3, 8, in_channel], "upconv_kernel")
        output_shape = [int(img_shape[0]), int(img_shape[1]), int(img_shape[2]), int(img_shape[3]), 8]
        output = tf.nn.conv3d_transpose(output, kernel, output_shape, [1, 2, 2, 2, 1])
        output = tf.concat([img, output], -1)
        print('output', output)
        with tf.variable_scope("conv_bf1"):
            _output = self._create_conv_layer(output)
            output = tf.concat([_output, output], -1)
        print('output', output)
        # with tf.variable_scope("conv_bf2"):
        #     _output = self._create_conv_layer(output)
        #     output = tf.concat([_output, output], -1)
        with tf.variable_scope("fc_layer"):
            output = self._create_conv_layer(output, 1, output_feature=2)
        print('output', output)

        with tf.variable_scope('prediction'):
            softmax_prob = tf.nn.softmax(logits=output, name='softmax_prob')
            predicted_label = tf.argmax(input=softmax_prob, axis=4, name='predicted_label')
        return output,predicted_label

    def loss_op(self, logits, labels, dst_weight=None):
        # TODO:remove receptive_field! or put it into config file.
        '''receptive_field = 0
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
        print(logits_slice)
        logits_slice = tf.squeeze(logits_slice)
        labels_slice = tf.squeeze(labels_slice)
        labels_slice = tf.cast(labels_slice, dtype=tf.float32)

        weight_loss = cfg.weight_loss
        pw = 1
        if weight_loss == 0:
            pw = (1 - tf.reduce_mean(labels_slice)) / tf.reduce_mean(labels_slice)
            # change it to calc weight.
        else:
            pw = weight_loss'''

        #loss = entropy_loss_function(logits, labels, dst_weight)
        with tf.name_scope("weighted_cross_entropy"):
            if cfg.use_jacc_loss == True:
                self.total_loss = jacc_loss(logits[0], labels)
            else:
                self.total_loss = softmax_loss_function(logits[0], labels, dst_weight)

        with tf.name_scope("accuracy"):
            if cfg.predict_op == 'sigmoid':
                correct_pred = tf.equal(tf.squeeze(tf.cast(logits[1] + 0.5, tf.int32),axis=-1), tf.cast(labels,dtype=tf.int32))
            else:
                #correct_pred = tf.equal(pred_labels,tf.cast(labels, dtype=tf.int64))
                correct_pred = tf.reduce_sum(logits[1]) / tf.reduce_sum(tf.cast(labels, dtype=tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('total loss', self.total_loss )

        return self.total_loss, accuracy

'''sigmoid_p = tf.nn.sigmoid(logits_slice)
# 新建一个与给定的tensor类型大小一致的tensor，其所有元素为0
zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

# tf.where(input, a,b)，其中a，b均为尺寸一致的tensor，作用是将a中对应input中true的位置的元素值不变，
# 其余元素进行替换，替换成b中对应位置的元素值

# For poitive prediction, only need consider front part loss, back part is 0;
# target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
pos_p_sub = tf.where(labels_slice > zeros, labels_slice - sigmoid_p, zeros)
alpha = 0.25
gamma = 2
# For negative prediction, only need consider back part loss, front part is 0;
# target_tensor > zeros <=> z=1, so negative coefficient = 0.
neg_p_sub = tf.where(labels_slice > zeros, zeros, sigmoid_p)
per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                      - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(
    tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))

loss = tf.reduce_mean(per_entry_cross_ent)'''
