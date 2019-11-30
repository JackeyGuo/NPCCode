import os
import random
import numpy as np
import tensorflow as tf
from tools import scorer
from tools import utils
from core import model_factory
from tools.timer import Timer
import config.config as cfg

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
        #self.leave_step = cfg.leave_step
        # record the min valid loss
        self.min_valid_loss = np.inf

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
        #print(variables)
        counter = 0
        for i in variables:
            shape = i.get_shape()
            val = 4
            for j in shape:
                val *= int(j)

            counter += val

        print("with: {0} trainables".format(counter))

        total_parameters = 0
        for variable in tf.trainable_variables():
            variable_parameters = 1
            for dim in variable.get_shape():
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Total number of trainable parameters: %d' % total_parameters)

    def apply_gradients(self, grads_and_vars):
        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            print(vars_with_grad)
            raise ValueError(
                "$$ ddpg $$ policy net $$ No gradients provided for any variable, check your graph for ops"
                " that do not support gradients,variables %s." %
                ([str(v) for _, v in grads_and_vars]))
        return self.optimizer.apply_gradients(grads_and_vars)

    def compute_gradients(self, total_loss):
        grads_and_vars = self.optimizer.compute_gradients(
            total_loss, var_list=tf.trainable_variables())
        grads = [g for (g, _) in grads_and_vars if g is not None]
        compute_op = tf.group(*grads)

        return (grads_and_vars, compute_op)

    def save_checkpoint(self, checkpoint_dir, model_name, global_step):
        model_dir = '%s_%s' % (self.batch_size, self.output_size)

        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
        # defaults to the list of all saveable objects

    def load_checkpoint(self, checkpoint_dir):
        model_dir = '%s_%s' % (self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
        # A CheckpointState if the state was available, None otherwise.
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint_state.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, checkpoint_name))
            return True
        else:
            return False

    def _get_optimizer(self,loss,opt_name="adam"):
        decay_steps = cfg.lr_decay_step

        decay_rate = cfg.lr_decay_rate

        if cfg.name == 'vnet' or cfg.name == 'vnet_new':
            learning_rate = 0.0001
        else:
            learning_rate = cfg.learn_rate

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

        '''update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            grads_and_vars, _ = self.compute_gradients(loss)

            optimizer_op = self.apply_gradients(grads_and_vars=grads_and_vars)'''

        self.optimizer = tf.train.AdamOptimizer(
            self.learning_rate).minimize(loss, global_step=self.global_step)
        '''num_update = tf.Variable(0, trainable=False)
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9999,num_updates=num_update)
        self.averages_op = self.ema.apply(tf.trainable_variables())
        with tf.control_dependencies([self.optimizer]):
            train_op = tf.group(self.averages_op)'''

        return self.optimizer

    def train(self, fold_num):

        train_holder,seg_holder,dst_holder = self.provider.get_train_holder()

        model = self.model_class(self.is_training)
        inference_op = model.inference_op(train_holder)

        if cfg.use_dst_weight == True:
            loss_op,acc_op = model.loss_op(inference_op, seg_holder, dst_holder)
        else:
            loss_op,acc_op = model.loss_op(inference_op, seg_holder)
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
        config = config
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)

            train_timer = Timer()
            load_timer = Timer()
            # if model checkpoint exist, then load last checkpoint
            #self._load_model(saver, sess, model_output_path)
            with open(file=loss_txt_path + '/loss_' + cfg.name + str(fold_num) + '.txt', mode='w', buffering=line_buffer) as loss_log:
                for step in range(self.train_step):

                    if cfg.use_dst_weight == True:
                        load_timer.tic()
                        image, label, weights = self.provider.get_train_value(with_weight=cfg.use_dst_weight)
                        image_val, label_val, val_weights = self.provider.get_val_value(with_weight=cfg.use_dst_weight)
                        load_timer.toc()

                        train_timer.tic()
                        train_merge, train_loss, _, train_acc = sess.run(
                            [merged, loss_op, train_op, acc_op],
                            feed_dict={train_holder: image, seg_holder: label, dst_holder: weights})
                        valid_merge, val_loss, val_acc = sess.run([merged, loss_op, acc_op],
                            feed_dict={train_holder: image_val,seg_holder: label_val, dst_holder: val_weights,self.is_training: False})
                        train_timer.toc()
                    else:
                        load_timer.tic()
                        image, label = self.provider.get_train_value(with_weight=cfg.use_dst_weight)
                        image_val, label_val = self.provider.get_val_value(with_weight=cfg.use_dst_weight)
                        load_timer.toc()

                        train_timer.tic()
                        train_merge, train_loss, _, train_acc = sess.run(
                            [merged, loss_op, train_op, acc_op],
                            feed_dict={train_holder: image, seg_holder: label})
                        valid_merge, val_loss, val_acc = sess.run(
                            [merged, loss_op, acc_op],
                            feed_dict={train_holder: image_val, seg_holder: label_val, self.is_training: False})
                        train_timer.toc()

                    #if val_loss < self.min_valid_loss:
                        #self.min_valid_loss = val_loss
                        #saver.save(sess, os.path.join(self.output_path, "model/model_%d_%.6f"%(fold_num,self.min_valid_loss)))
                    if np.mod(step + 1, self.save_interval) == 0:
                        #saver_final = tf.train.Saver(max_to_keep=1)
                        saver.save(sess, os.path.join(self.output_path,"model/model_saved_%d"%fold_num))
                        #saver_final.save(sess, os.path.join(self.output_path, "model_final/model_saved_%d"%fold_num))
                    '''train_merge, train_loss, t_dice_loss, t_weight_loss, m_dice_loss, m_weight_loss,_ = sess.run([merged,
                                                        loss_op, total_dice_loss, total_weight_loss,
                                                        main_dice_loss, main_weight_loss,train_op],
                                                       feed_dict={train_holder: image, seg_holder: label})'''
                    '''train_merge, train_loss, t_dice_loss, t_focal_loss, m_dice_loss, m_focal_loss, _ = sess.run(
                        [merged,
                         loss_op, total_dice_loss, total_focal_loss,
                         main_dice_loss, main_focal_loss, train_op],
                        feed_dict={train_holder: image, seg_holder: label})'''
                    '''train_merge, train_loss, t_dice_loss, m_dice_loss, _ = sess.run(
                        [merged,
                         loss_op, total_dice_loss,
                         main_dice_loss, train_op],
                        feed_dict={train_holder: image, seg_holder: label})'''

                    '''output_format = '[Epoch]%d, Speed: %.3fs/iter,Load: %.3fs/iter, Remain: %s' \
                                    ' train_loss: %.8f, valid_loss: %.8f\n' \
                                    '[Loss]dice_loss: %.8f,main_dice_loss: %.8f \n' \
                                    % (step, train_timer.average_time, load_timer.average_time,
                                       train_timer.remain(step, self.train_step), train_loss, val_loss,
                                       t_dice_loss, m_dice_loss)'''
                    '''output_format = '[Epoch]%d, Speed: %.3fs/iter,Load: %.3fs/iter, Remain: %s' \
                                    ' train_loss: %.8f, valid_loss: %.8f\n' \
                                    '[Loss]dice_loss: %.8f, focal_loss: %.8f, main_dice_loss: %.8f, main_focal_loss: %.8f\n' \
                                    % (step, train_timer.average_time, load_timer.average_time,
                                       train_timer.remain(step,self.train_step),train_loss, val_loss,
                                       t_dice_loss, t_focal_loss, m_dice_loss, m_focal_loss)'''
                    '''output_format = 'Epoch:%d,Speed: %.3fs/iter,Load: %.3fs/iter,Remain: %s\n'\
                                    'train_loss: %.8f,valid_loss: %.8f,main_dice_loss: %.8f,main_weight_loss: %.8f'\
                                    % (step, train_timer.average_time, load_timer.average_time,
                                       train_timer.remain(step, self.train_step), train_loss, val_loss,
                                       m_dice_loss, m_weight_loss)'''
                    '''output_format = '[Epoch]%d, Speed: %.3fs/iter,Load: %.3fs/iter, Remain: %s' \
                                    ' train_loss: %.8f, valid_loss: %.8f\n' \
                                    '[Loss] main_jacc_loss: %.8f, auxi_jacc_loss: %.8f\n' \
                                    % (step, train_timer.average_time, load_timer.average_time,
                                       train_timer.remain(step, self.train_step), train_loss, val_loss,
                                       main_jacc_loss, auxi_jacc_loss)'''
                    output_format = "train loss: %f, valid loss: %f, train accuracy: %f, val accuracy: %f, step: %d" % \
                                    (train_loss, val_loss, train_acc, val_acc, step)
                    print(output_format)
                    train_writer.add_summary(train_merge, step)
                    test_writer.add_summary(valid_merge, step)

                    if step % 5 == 0:
                        loss_log.write(output_format + '\n')
                    #if np.mod(step + 1, self.save_interval) == 0:
                        #saver.save(sess, os.path.join(self.output_path, "model/model_saved_%d"%fold_num))
                train_writer.close()
                test_writer.close()

    def predict(self,fold_num):
        tf.reset_default_graph()

        is_training = tf.Variable(tf.constant(False))
        test_holder = self.provider.get_test_holder()

        model = self.model_class(is_training)
        if cfg.name == 'vnet' or cfg.name == 'vnet_new' or cfg.name == 'unet' or cfg.name == 'cnn_v2' \
                or cfg.name == 'denseDilatedASPP' or cfg.name == 'deeplab':
            _,predict_label = model.inference_op(test_holder)

        else:
            _,predict_label,_,_,_ = model.inference_op(test_holder)
        with tf.Session() as sess:
            # TODO: load pre-trained model
            # TODO: load checkpoint
            saver = tf.train.Saver()
            #saver.restore(sess, os.path.join(self.output_path, "model/model_%d_%.6f"%(fold_num,self.min_valid_loss)))
            saver.restore(sess, os.path.join(self.output_path, "model/model_saved_%d"%fold_num))

            self.provider.init_test()
            while True:
                test_lst = self.provider.get_test_value()
                if test_lst is None:
                    break
                output_lst = []
                for list in test_lst:
                    output = sess.run(predict_label, feed_dict={test_holder: list})
                    output_lst.append(output)
                self.provider.write_test(output_lst)

def main():

    if cfg.gpu:
        gpu = cfg.gpu
    else:
        gpu = ''
    data_npy_path = os.path.join(cfg.output_path, "data")
    if not os.path.exists(data_npy_path):
        os.makedirs(data_npy_path)

    # cross_validation is leave_one_out or k_fold
    cross_validation = 'k_fold'
    fold_num = cfg.fold_num
    count = 1
    one_by_one = cfg.one_by_one
    # set cuda visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    model = model_factory.modelMap[cfg.name]
    print(cfg.name)
    provider = utils.DataProvider()

    if cross_validation == 'leave_one_out':
        size = len(patient_list)
        val_size = (size - cfg.leave_step) // 10

        pat_iter = 0
        random.shuffle(patient_list)
        while True:
            if pat_iter == len(patient_list):
                break
            elif pat_iter + cfg.leave_step < len(patient_list):
                # split train val test data, train : 86, val : 9, test : 25
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
    else:
        if one_by_one == False:
            patient_list = provider.getPatientName()
            val_size = len(patient_list) // fold_num

            pat_iter = 0
            random.shuffle(patient_list)
            # write patient_list to npy
            np.save(data_npy_path + '/data.npy', patient_list)
        else:
            data_path = os.path.join(cfg.output_path,"data/data.npy")
            data = np.load(data_path)
            print('load data successfully!!!!!!!!!!!')
            patient_list = []
            for i in range(len(data)):
                patient_list.append(data[i])
            val_size = len(patient_list) // fold_num
            #fold_iter = cfg.fold_iter
            pat_iter = (cfg.begin_fold_iter - 1) * val_size
            count = cfg.begin_fold_iter
            end_iter = (cfg.end_fold_iter) * val_size
        print(len(patient_list), cross_validation, fold_num)
        while True:
            if cfg.one_by_one == False and pat_iter == len(patient_list):
                break
            elif cfg.one_by_one == True and pat_iter == end_iter:
                break
            else:
                # split train val test data, train : 96, val : 24
                val_set = patient_list[pat_iter:pat_iter + val_size]
                train_set = patient_list[:pat_iter] + patient_list[pat_iter + val_size:]

                provider.setTrainVal(train_set, val_set)
                tf.reset_default_graph()

                processor = Solver(model, provider)

                processor.train(count)
                processor.predict(count)
                count += 1
                pat_iter += val_size

    # caculate score of dice
    truth_path = cfg.data_path
    #output_path = r"/home/data_new/guofeng/projects/Segmentation/NPCCode/result/%s/score" % (cfg.name)
    output_path = os.path.join(cfg.output_path, "score")
    #predict_path = r"/home/data_new/guofeng/projects/Segmentation/NPCCode/result/%s_sqrt/predict" % (cfg.name)
    predict_path = os.path.join(cfg.output_path, "predict")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    tpo = scorer.create_t_p_o_list(truth_path, predict_path, output_path)
    scorer.score(tpo)

if __name__ == "__main__":
    main()