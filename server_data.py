import argparse
import os
import multiprocessing as mp
import socket
import numpy as np

from tensorpack import *
from tensorpack.dataflow import send_dataflow_zmq, MapData, TestDataSpeed, FakeData
from tensorpack.utils import logger

from zmq_ops import dump_arrays


def get_data(train_or_test,batch):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    #pp_mean = ds.get_per_pixel_mean()
    pc_mean = np.array([125.3, 123.0, 113.9])
    pc_std = np.array([63.3, 62.1, 66.7])
    parallel = min(40, mp.cpu_count())
    if isTrain:
        augmentors = [
            imgaug.MapImage(lambda x: (x - pc_mean)/(pc_std)),
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: (x - pc_mean)/(pc_std))
            #imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    if isTrain:
    	ds = PrefetchDataZMQ(ds, parallel)
    	ds = BatchData(ds, batch, remainder=False)
    return ds




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--fake', action='store_true')
    parser.add_argument('--batch', help='per-GPU batch size',
                        default=32, type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--gpu',default='0', help='comma separated list of GPU(s) to use.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.fake:
    	pass
    else:
        ds_train = get_data('train', args.batch)
        ds_test = get_data('test', args.batch)

    logger.info("Running on {}".format(socket.gethostname()))

    if args.benchmark:
        ds_train = MapData(ds_train, dump_arrays)
        TestDataSpeed(ds_train, warmup=300).start()
    else:
        send_dataflow_zmq(
            ds_train, 'ipc://@Cifar10-train-b{}'.format(args.batch),hwm=150, format='zmq_ops', bind=True)
        
        send_dataflow_zmq(
            ds_test, 'ipc://@Cifar10-test-b{}'.format(args.batch),hwm=150, format='zmq_ops', bind=True)