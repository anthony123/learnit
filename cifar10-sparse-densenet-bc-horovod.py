#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os
import socket


from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.callbacks import *
import horovod.tensorflow as hvd

#os.environ['TENSORPACK_DATASET'] = '/media/liuwq/hd/dataset/tensorpack_data'

"""
CIFAR10 DenseNet example. See: http://arxiv.org/abs/1608.06993
Code is developed based on Yuxin Wu's ResNet implementation: https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet
Results using DenseNet (L=40, K=12) on Cifar10 with data augmentation: ~5.77% test error.

Running time:
On one TITAN X GPU (CUDA 7.5 and cudnn 5.1), the code should run ~5iters/s on a batch size 64.
"""

BATCH_SIZE = 64

class Model(ModelDesc):
    def __init__(self,k, path,num_block1, num_block2, num_block3,num_block4,loss_scale=1.0):
        super(Model, self).__init__()
        #self.N = int(layers_per_block)
        self.growthRate = int(k)
        self.num_path = int(path)
        self.input_channel = 2*self.growthRate
        self.num_block1 = int(num_block1)
        self.num_block2 = int(num_block2)
        self.num_block3 = int(num_block3)
        self.num_block4 = int(num_block4)
        self._loss_scale = loss_scale

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')
               ]

    def _build_graph(self, input_vars):
        super(Model, self)._build_graph(input_vars)
        image, label = input_vars

        #if self._loss_scale != 1.0:
            #self.cost = self.cost * self._loss_scale
        #image = image / 128.0 - 1

        
        
        def conv(name, l, channel, stride, nl=tf.identity):
            return Conv2D(name, l, channel, 3, stride=stride,
                          nl=nl, use_bias=True,
                          W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/channel)))
        def add_layer(name, l, block_idx):
            with tf.variable_scope(name) as scope:
                shape = l.get_shape().as_list()
                in_channel = shape[3]
                curr_growthRate = int(self.growthRate)
                curr_input_channel = int(self.input_channel)
                curr_num_block = (in_channel-curr_input_channel)/curr_growthRate + 1

                with tf.variable_scope('bn2'):
                    c = tf.contrib.layers.batch_norm(l, decay=0.9, scale=True, is_training = get_current_tower_context().is_training, updates_collections=None, reuse=None)
                #c = tf.nn.relu(c)
                #c = conv('conv2', c, curr_growthRate, 1)
                #bc_channel = (curr_num_block+1)//2*curr_growthRate
                #bc_channel = curr_growthRate//2*curr_num_block
                bc_channel = 4*self.growthRate

                # if block_idx == 1:
                #     bc_channel = 12
                # elif block_idx == 2:
                #     bc_channel = 24
                c = Conv2D('conv2', c, bc_channel , 1, stride=1, use_bias=True, nl=tf.identity)


                #c = BatchNorm('bn1', l)
                with tf.variable_scope('bn1'):
                    c = tf.contrib.layers.batch_norm(c, decay=0.9, scale=True, is_training = get_current_tower_context().is_training, updates_collections=None, reuse=None)
                c = tf.nn.relu(c)
                c = conv('conv1', c, curr_growthRate, 1)

                # print('curr_growthRate: %d'  %(curr_growthRate))
                # print('curr_input_channel: %d' %(curr_input_channel))
                # print('in_channel %d'  %(in_channel))
                # print('curr_num_block %d' %((in_channel-curr_input_channel)/curr_growthRate + 1))
                if(in_channel-curr_input_channel)%curr_growthRate != 0:
                    return 
                
                if curr_num_block > self.num_path:
                    split1, _, split2 = tf.split(l, [int(round(curr_num_block/2*curr_growthRate)), int(curr_growthRate), int(in_channel-curr_growthRate -round(curr_num_block/2*curr_growthRate))],3)
                    l = tf.concat([c, split1, split2],3)
                else:
                    l = tf.concat([c, l], 3)
            
            return l

        def add_transition(name, l, idx):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            curr_growthRate = int(self.growthRate)
            curr_input_channel = int(self.input_channel)
            curr_num_block = (in_channel-curr_input_channel)/curr_growthRate + 1
            next_growthRate = int(self.growthRate)
            next_input_channel = int(self.input_channel)
            
            out_channel=0
            if curr_num_block%2 == 0:
                out_channel = next_input_channel + (curr_num_block)*next_growthRate//2
            else:
                out_channel = next_input_channel + (curr_num_block-1)*next_growthRate//2

            with tf.variable_scope(name) as scope:
                #l = BatchNorm('bn1', l)
                with tf.variable_scope('bn1'):
                    l = tf.contrib.layers.batch_norm(l, decay=0.9, scale=True, is_training = get_current_tower_context().is_training, updates_collections=None, reuse=None)
                l = tf.nn.relu(l)
                l = Conv2D('conv1', l, out_channel, 1, stride=1, use_bias=True, nl=tf.identity)
                l = AvgPooling('pool', l, 2)
            return l


        def dense_net(name):
            total = 0
            l = conv('conv0', image, self.input_channel, 1, nl=tf.nn.relu)
            with tf.variable_scope("blcok1") as scope:
                for i in range(self.num_block1):
                    l = add_layer('dense_layer.{}'.format(i), l, 0)
            l = add_transition("trasition1", l, 0)
            with tf.variable_scope("block2") as scope:
                for i in range(self.num_block2):
                    l = add_layer('dense_layer.{}'.format(i), l, 1)
            l = add_transition("trasition2", l, 1)
            with tf.variable_scope("block3") as scope:
                for i in range(self.num_block3):
                    l = add_layer('dense_layer.{}'.format(i), l, 2)
            #l = BatchNorm('bnlast', l)
            with tf.variable_scope('bnlast'):
                l = tf.contrib.layers.batch_norm(l, decay=0.9, scale=True, is_training = get_current_tower_context().is_training, updates_collections=None, reuse=None)
            l = tf.nn.relu(l)
            l = GlobalAvgPooling('gap', l)
            logits = FullyConnected('linear', l, out_dim=10, nl=tf.identity)

            return logits


        logits = dense_net("dense_net")

        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label)
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W
        #wd_cost = tf.multiply(1e-4, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        wd_cost = tf.multiply(1e-4, l2, name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_config(model, fake=False):
    log_dir = 'train_log/cifar10-bc-k[%d]-path[%d]-[%d-%d-%d-%d]-' % ( int(args.k), int(args.path), int(args.block1), int(args.block2),int(args.block3), int(args.block4))
    logger.set_logger_dir(log_dir, action='n')

    # prepare dataset
    #dataset_train = get_data('train')
    
    #dataset_test = get_data('test')
    
    batch = args.batch
    total_batch = batch * hvd.size()
    if fake:
        pass
    else:
        logger.info("#Tower: {}; Batch size per tower: {}".format(hvd.size(), batch))
        data_train = ZMQInput('ipc://@Cifar10-train-b{}'.format(batch), 30, bind=False)
        data_test = ZMQInput('ipc://@Cifar10-test-b{}'.format(batch), 30, bind=False)
        data_train = StagingInput(data_train, nr_stage=1)
        data_test = StagingInput(data_test, nr_stage=1)
        steps_per_epoch = 781
        
    BASE_LR = 0.1 * (total_batch // BATCH_SIZE)
    logger.info("Base LR: {}".format(BASE_LR))
    callbacks = [
        ModelSaver(max_to_keep=100),
        ScheduledHyperParamSetter(
            'learning_rate', [(args.drop_1, BASE_LR * 1e-1), (args.drop_2, BASE_LR * 1e-2),
                              (args.drop_3, BASE_LR * 5e-3)]),
    ]

    #!!MAKE SURE THE TOTAL BATCH SIZE IS EQUAL TO THE BATCH_SIZE, AND THE FOLLOWING CODE DOES NOT Running
    assert total_batch == BATCH_SIZE
    if BASE_LR != 0.1:
        """
        Sec 2.2: In practice, with a large minibatch of size kn, we start from a learning rate of η and increment
        it by a constant amount at each iteration such that it reachesη = kη after 5 epochs. After the warmup phase, we go back
        to the original learning rate schedule.
        """
        callbacks.append(
            ScheduledHyperParamSetter(
                'learning_rate', [(0, 0.1), (5 * steps_per_epoch, BASE_LR)],
            interp='linear', step_based=True))
        #ADD test CODE

    if hvd.rank() == 0 and not args.fake:
        # TODO For distributed training, you probably don't want everyone to wait for validation.
        # Better to start a separate job, since the model is saved.
        pass

    return TrainConfig(
        model=model,
        data=data_train,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        max_epoch=35 if args.fake else args.max_epoch,
        )


if __name__ == '__main__':
    #BATCH_SIZE = 64
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',default='0', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    parser.add_argument('--drop_1',default=150, help='Epoch to drop learning rate to 0.01.') # nargs='*' in multi mode
    parser.add_argument('--drop_2',default=200,help='Epoch to drop learning rate to 0.001')
    parser.add_argument('--drop_3',default=250,help='Epoch to drop learning rate to 0.0002')
    parser.add_argument('--max_epoch',default=280,help='max epoch')
    parser.add_argument('--k', default=15, help='number of output feature maps for each dense block')
    parser.add_argument('--path', default=10,help='number of paths to each layer')#12
    parser.add_argument('--fake', help='use fakedata to test or benchmark this model', action='store_true')
    parser.add_argument('--block1', default=10, help='number of layers  for the first block') 
    parser.add_argument('--block2', default=15, help='number of layers  for the second block')   
    parser.add_argument('--block3', default=20, help='number of layers  for the third block')   
    parser.add_argument('--block4', default=0,  help='number of layers  for the fourth block')  
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--batch', help='per-GPU batch size', default=32, type=int)


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.info("Running on {}".format(socket.gethostname()))
    hvd.init()
    if hvd.rank() == 0:
        #logger.set_logger_dir(args.logdir, 'd')
        pass


    logger.info("hvd: ".format(hvd.size()))
    model=Model(args.k, args.path, args.block1, args.block2, args.block3,args.block4,loss_scale=1.0 / hvd.size())
    config = get_config(model, fake=args.fake)
    trainer = HorovodTrainer(average=False)
    launch_train_with_config(config, trainer)