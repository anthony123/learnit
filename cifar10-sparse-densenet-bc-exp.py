#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os


from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.train import TrainConfig, SyncMultiGPUTrainerParameterServer, DistributedTrainerReplicated
from tensorpack.tfutils.summary import *
from tensorpack.tfutils import argscope, get_model_loader, get_default_sess_config
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.callbacks import *

#os.environ['TENSORPACK_DATASET'] = '/media/liuwq/hd/dataset/tensorpack_data'
os.environ['TENSORPACK_TRAIN_API'] = 'v1'

"""
CIFAR10 DenseNet example. See: http://arxiv.org/abs/1608.06993
Code is developed based on Yuxin Wu's ResNet implementation: https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet
Results using DenseNet (L=40, K=12) on Cifar10 with data augmentation: ~5.77% test error.

Running time:
On one TITAN X GPU (CUDA 7.5 and cudnn 5.1), the code should run ~5iters/s on a batch size 64.
"""

#BATCH_SIZE = 64

class Model(ModelDesc):
    def __init__(self,k, path,num_block1, num_block2, num_block3,num_block4):
        super(Model, self).__init__()
        #self.N = int(layers_per_block)
        self.growthRate = int(k)
        self.num_path = int(path)
        self.input_channel = 2*self.growthRate
        self.num_block1 = int(num_block1)
        self.num_block2 = int(num_block2)
        self.num_block3 = int(num_block3)
        self.num_block4 = int(num_block4)

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')
               ]

    def _build_graph(self, input_vars):
        image, label = input_vars
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

                with tf.variable_scope('bn2') as scope:
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
                    c = tf.contrib.layers.batch_norm(c, decay=0.9, scale=True, is_training = get_current_tower_context().is_training, updates_collections=None)
                c = tf.nn.relu(c)
                c = conv('conv1', c, curr_growthRate, 1)

                # print('curr_growthRate: %d'  %(curr_growthRate))
                # print('curr_input_channel: %d' %(curr_input_channel))
                # print('in_channel %d'  %(in_channel))
                # print('curr_num_block %d' %((in_channel-curr_input_channel)/curr_growthRate + 1))
                if(in_channel-curr_input_channel)%curr_growthRate != 0:
                    return 
                
                if curr_num_block > self.num_path:
                    split1, _, split2 = tf.split(l, [curr_num_block/2*curr_growthRate, curr_growthRate, in_channel-curr_growthRate -curr_num_block/2*curr_growthRate],3)
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
            #out_channel = next_input_channel + (curr_num_block-1)*next_growthRate
            #out_channel = next_input_channel
            #out_channel = in_channel
            # print("next_input_channel: %d"  %(next_input_channel))
            # print("next_growthRate: %d"  %(next_growthRate))
            #print("curr_num_block: %d" %(curr_num_block))
            #print("in_channel: %d" %(in_channel))
            # print("curr_growthRate: %d" %(curr_growthRate))
            # print("curr_input_channel: %d" %(curr_input_channel))
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


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    #pp_mean = ds.get_per_pixel_mean()
    pc_mean = np.array([125.3, 123.0, 113.9])
    pc_std = np.array([63.3, 62.1, 66.7])
    if isTrain:
        augmentors = [
            imgaug.MapImage(lambda x: (x - pc_mean)/(pc_std)),
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            #imgaug.Brightness(20),
            #imgaug.Contrast((0.6,1.4)),
            #imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: (x - pc_mean)/(pc_std))
            #imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    nr_tower = args.gpu.split(',')
    BATCH_SIZE = 64
    BATCH_SIZE  /= len(nr_tower)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds

def get_config():
    log_dir = 'train_log/cifar10-bc-k[%d]-path[%d]-[%d-%d-%d-%d]-' % ( int(args.k), int(args.path), int(args.block1), int(args.block2),int(args.block3), int(args.block4))
    logger.set_logger_dir(log_dir, action='n')

    # prepare dataset
    dataset_train = get_data('train')
    
    dataset_test = get_data('test')
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    # config = tf.ConfigProto(allow_soft_placement = True,gpu_options=gpu_options)
    # config.gpu_options.allow_growth=True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    callbacks = []
    callbacks.append(ModelSaver())
    nr_tower = len(args.gpu.split(','))
    print('nr_tower = {}'.format(nr_tower))
    steps_per_epoch = dataset_train.size()//nr_tower
    if nr_tower == 1:
            # single-GPU inference with queue prefetch
        callbacks.append(InferenceRunner(dataset_test,
                [ScalarStats('cost'), ClassificationError()]))
    else:
        # multi-GPU inference (with mandatory queue prefetch)
        callbacks.append(DataParallelInferenceRunner(
                dataset_test, [ScalarStats('cost'), ClassificationError()], list(range(nr_tower))))
        #callbacks.append(InferenceRunner(dataset_test,
                #[ScalarStats('cost',prefix="testing"), ClassificationError(summary_name='validataion_error1')]))

    # callbacks.append(DataParallelInferenceRunner(
    #             dataset_test, [ScalarStats('cost'), ClassificationError()], list(range(nr_tower))))
    callbacks.append(ScheduledHyperParamSetter('learning_rate', [(0, 0.1), (args.drop_1, 0.01), (args.drop_2, 0.001),(args.drop_3, 0.0002)]))
    return TrainConfig(
        dataflow=dataset_train,
        # callbacks=[
        #     ModelSaver(),
        #     InferenceRunner(dataset_test,
        #         [ScalarStats('cost'), ClassificationError()]),
        #     ScheduledHyperParamSetter('learning_rate',
        #                               [(1, 0.1), (args.drop_1, 0.01), (args.drop_2, 0.001),(args.drop_2, 0.0001)])
        # ],
        callbacks=callbacks,
        model=Model(args.k, args.path, args.block1, args.block2, args.block3,args.block4),
        steps_per_epoch=steps_per_epoch,
        max_epoch=args.max_epoch,
        #session_config = config,
        nr_tower=nr_tower,

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
    parser.add_argument('--k', default=25, help='number of output feature maps for each dense block')
    parser.add_argument('--path', default=90,help='number of paths to each layer')#12
    
    parser.add_argument('--block1', default=30, help='number of layers  for the first block') 
    parser.add_argument('--block2', default=45, help='number of layers  for the second block')   
    parser.add_argument('--block3', default=60, help='number of layers  for the third block')   
    parser.add_argument('--block4', default=0,  help='number of layers  for the fourth block')  
    parser.add_argument('--job', type=str, help='job type (ps or worker)')
    parser.add_argument('--task', type=int, help='task id')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpu_options = tf.GPUOptions(allow_growth=True)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    config = get_config()
    config.data = QueueInput(config.dataflow)
    #assert config.data
    assert config.model
    #config.gpu_options.allow_growth = True
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
        #config.nr_tower=max(get_nr_gpu(), 1),
    #SyncMultiGPUTrainer(config).train()
    #SyncMultiGPUTrainerParameterServer(config).train()
    hosts = ['gn07', 'gn08']
    cluster_spec = tf.train.ClusterSpec({'ps': [h + ':2222' for h in hosts],'worker': [h + ':2223' for h in hosts]})
    server = tf.train.Server(cluster_spec, job_name=args.job, task_index=args.task,config = get_default_sess_config())
    #trainer = SyncMultiGPUTrainerParameterServer(len(args.gpu.split(',')))
    #launch_train_with_config(config, trainer)
    #launch_train_with_config(config, DistributedTrainerReplicated(4, server))
    #launch_train_with_config(config, DistributedTrainerReplicated(config.nr_tower, server))
    DistributedTrainerReplicated(config, server).train()

# ds = get_data('train')
# #print(help(ds.get_data()))

# #print(ds.size())
# #print(ds.get_data().next())

# #print(ds.get_data().size())

# sz = ds.size();

# for _ in range(sz):
#     print(ds.get_data().next())
