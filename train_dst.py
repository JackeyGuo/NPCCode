import os
import random

import numpy as np
import tensorflow as tf

from config import config_dst as cfg
from core import model_factory
from tools import scorer
from tools import utils
from tools.timer import Timer


class Solver(object):
    """
    this class is the base processor.
    """
    def __init__(self,model_class,provider):
        """
        :param model_class:the Class of model.for pass the is_training.
        :param provider: the instance of the provider.
        """
        self.is_training = tf.Variable(tf.constant(True,tf.bool))
        self.model_class = model_class
        self.model_name = cfg.name
        self.provider = provider
        self.train_step = cfg.iter_step
        self.output_path = cfg.output_path
        #self.record_step = cfg.record_step
        self.leave_step = cfg.leave_step

        #self.saver = tf.train.Saver(max_to_keep=3)
        self.save_interval = cfg.save_interval

        self.output_channels = 1

        self.batch_size =1
        self.output_size = 1

    def _load_model(self, saver, sess, model_path):

        latest_checkpoint = tf.train.latest_checkpoint(model_path)
        #print(latest_checkpoint)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
            print("Checkpoint loaded")
        else:
            print("First time to train!")

    def _count_trainables(self):
        variables = tf.trainable_variables()
        counter = 0
        for i in variables:
            shape = i.get_shape()
            val = 4
            for j in shape:
                val *= int(j)

            counter += val

        print("with: {0} trainables".format(counter))

    def _get_optimizer(self,loss,opt_name="adam"):
        decay_steps = cfg.lr_decay_step
        learning_rate = cfg.learn_rate
        decay_rate = cfg.lr_decay_rate

        #trainable = tf.trainable_variables()
        self.global_step = tf.get_variable(
            'global_step', [], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
        # Decay the learning rate exponentially based on the number of steps.
        self.learning_rate = tf.train.exponential_decay(learning_rate,
                                        self.global_step,
                                        decay_steps,
                                        decay_rate,
                                        staircase=True,
                                        name='learning_rate')
        tf.summary.scalar('learning_rate', self.learning_rate)

        self.optimizer = tf.train.AdamOptimizer(
            self.learning_rate).minimize(loss, global_step=self.global_step)

        return self.optimizer

    def train(self):
        """
        now tf_records are no used for the full image.
        :return:
        """

        train_holder,seg_holder,dst_holder = self.provider.get_train_holder()

        if self.model_name == 'cnn_v2':
            model = self.model_class(self.is_training)
            model.build_model(train_holder, seg_holder)
            total_loss = model.total_loss
            total_dice_loss = model.total_dice_loss
            total_weight_loss = model.total_weight_loss
            #main_dice_loss = model.main_dice_loss
            #dice = model.dice_coefficient

            loss_op = model.entropy_loss
            train_op = self._get_optimizer(total_loss)
        else:
            model = self.model_class(self.is_training)
            inference_op = model.inference_op(train_holder)

            if cfg.use_dst_weight == True:
                loss_op = model.loss_op(inference_op, seg_holder, dst_holder)
            else:
                loss_op = model.loss_op(inference_op, seg_holder)
            #loss_op = model.loss_op(inference_op, seg_holder)

            total_dice_loss = model.total_dice_loss
            total_weight_loss = model.total_weight_loss
            main_weight_loss = model.main_weight_loss
            main_dice_loss = model.main_dice_loss

            train_op = self._get_optimizer(loss_op)

        merged = tf.summary.merge_all()
        self._count_trainables()
        log_output_path = os.path.join(self.output_path,"log")
        if not os.path.exists(log_output_path):
            os.makedirs(log_output_path)

        model_output_path = os.path.join(self.output_path,"model")
        if not os.path.exists(model_output_path):
            os.makedirs(model_output_path)

        loss_txt_path = os.path.join(self.output_path, "loss")
        if not os.path.exists(loss_txt_path):
            os.makedirs(loss_txt_path)

        train_writer = tf.summary.FileWriter(os.path.join(log_output_path, "train"))
        test_writer = tf.summary.FileWriter(os.path.join(log_output_path, "val"))

        line_buffer = 1
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            train_timer = Timer()
            load_timer = Timer()

            # if model checkpoint exist, then load last checkpoint
            #self._load_model(saver, sess, model_output_path)

            with open(file=loss_txt_path + '/loss_' + cfg.name + '.txt', mode='w', buffering=line_buffer) as loss_log:
                for step in range(self.train_step):
                    load_timer.tic()
                    image, label, weight = self.provider.get_train_value(with_weight=cfg.use_weight)
                    image_val, label_val, weight = self.provider.get_val_value(with_weight=cfg.use_weight)
                    load_timer.toc()

                    train_timer.tic()
                    train_merge, train_loss, t_dice_loss, t_weight_loss, m_dice_loss, m_weight_loss,_ = sess.run([merged,
                                                        loss_op, total_dice_loss, total_weight_loss,
                                                        main_dice_loss, main_weight_loss,train_op],
                                                       feed_dict={train_holder: image, seg_holder: label, dst_holder: weight})
                    valid_merge, val_loss = sess.run([merged, loss_op],
                                                    feed_dict={train_holder: image_val,
                                                    seg_holder: label_val, dst_holder: weight, self.is_training: False})
                    train_timer.toc()
                    output_format = '[Epoch]%d, Speed: %.3fs/iter,Load: %.3fs/iter, Remain: %s' \
                                    ' train_loss: %.8f, valid_loss: %.8f\n' \
                                    '[Loss]dice_loss: %.8f, weight_loss: %.8f, main_dice_loss: %.8f, main_weight_loss: %.8f\n' \
                                    % (step, train_timer.average_time, load_timer.average_time,
                                       train_timer.remain(step,self.train_step),train_loss, val_loss,
                                       t_dice_loss, t_weight_loss, m_dice_loss, m_weight_loss)
                    print(output_format)
                    train_writer.add_summary(train_merge, step)
                    test_writer.add_summary(valid_merge, step)

                    if step % 10 == 0:
                        loss_log.write('train loss: %.5f, valid_loss: %.5f, glabl step: %d' % (train_loss,val_loss,step) + '\n')

                    if np.mod(step + 1, self.save_interval) == 0:
                        saver.save(sess, os.path.join(self.output_path, "model/model_saved"))
                train_writer.close()
                test_writer.close()

    def predict(self):
        tf.reset_default_graph()

        is_training = tf.Variable(tf.constant(False))
        test_holder = self.provider.get_test_holder()

        model = self.model_class(is_training)
        if cfg.name == 'cnn':

            output_prob, output_label, _, _, _ = model.unet_model(test_holder)
            output_op = tf.sigmoid(output_prob)

        else:
            output_op,_,_,_ = model.inference_op(test_holder)
            predict_op = cfg.predict_op

            if predict_op == "sigmoid":
                output_op = tf.nn.sigmoid(output_op)

        #model_output_path = os.path.join(self.output_path, "model")
        with tf.Session() as sess:
            # TODO: load pre-trained model
            # TODO: load checkpoint
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(self.output_path, "model/model_saved"))

            self.provider.init_test()
            while True:
                test_lst = self.provider.get_test_value()
                if test_lst is None:
                    break
                output_lst = []
                for list in test_lst:
                    output = sess.run(output_op, feed_dict={test_holder: list})
                    output_lst.append(output)
                self.provider.write_test(output_lst)

def main():

    if cfg.gpu:
        gpu = cfg.gpu
    else:
        gpu = '0'

    # set cuda visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    model = model_factory.modelMap[cfg.name]
    print(cfg.name)
    provider = utils.DataProvider()
    patient_list = provider.getPatientName()
    size = len(patient_list)
    val_size = (size - cfg.leave_step) // 10

    pat_iter = 0
    random.shuffle(patient_list)
    while True:
        if pat_iter == len(patient_list):
            break
        elif pat_iter + cfg.leave_step < len(patient_list):
            # split train val test data
            test_set = patient_list[pat_iter:pat_iter + cfg.leave_step]
            train_set = patient_list[:pat_iter] + patient_list[pat_iter + cfg.leave_step:]

            val_set = train_set[-val_size:]
            train_set = train_set[:-val_size]
            provider.setTrainValTest(train_set, val_set, test_set)
            tf.reset_default_graph()

            processor = Solver(model,provider)
            processor.train()
            processor.predict()
            pat_iter += cfg.leave_step
        else:
            test_set = patient_list[pat_iter:pat_iter + cfg.leave_step]
            train_set = patient_list[:pat_iter] + patient_list[pat_iter + cfg.leave_step:]
            tf.reset_default_graph()

            val_set = train_set[-val_size:]
            train_set = train_set[:-val_size]
            provider.setTrainValTest(train_set, val_set, test_set)

            processor = Solver(model,provider)
            processor.train()
            processor.predict()
            pat_iter = len(patient_list)

    # caculate score of dice
    output_path = r"/home/data_new/guofeng/projects/Segmentation/NPCCode/result/%s/score" % cfg.name
    truth_path = r"/home/gf/guofeng/0120/all_resized_124"
    predict_path = r"/home/data_new/guofeng/projects/Segmentation/NPCCode/result/%s/predict" % cfg.name
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    tpo = scorer.create_t_p_o_list(truth_path, predict_path, output_path)
    scorer.score(tpo)

if __name__ == "__main__":
    main()