import argparse
import logging
import pprint
import mxnet as mx
import numpy as np
import time

#from rcnn.config import config, default, generate_config
#from rcnn.symbol import *

def parse_args():
    parser = argparse.ArgumentParser(description='Test kvstore speed')

    args, rest = parser.parse_known_args()

    # basic
    #parser.add_argument('--frequent', help='frequency of logging', default=default.frequent, type=int)
    parser.add_argument('--kvstore', help='the kv-store type', default=None, type=str)
    parser.add_argument('--work_load_list', help='work load for different devices', default=None, type=list)
    # e2e
    parser.add_argument('--gpus', help='GPU device to train with', default='0', type=str)
    args = parser.parse_args()
    return args

def kvtest(args):
    print ("args:")
    #create the distributed kv store on device or not
    kv=mx.kvstore.create(args.kvstore)

    #initialize kv store
    size_para=500       #size of parameter in MB
    ini_array=mx.nd.full((size_para,262144),1.0)
    print("handle before push pull",type(ini_array.handle))

    key=[ str(i) for i in range(1,2)]        #key store
    kv.init(key[0],ini_array)              #initialize with 1.0
    kv.set_optimizer(mx.optimizer.SGD())   #set merge method

    #define push data
    grad=mx.nd.full((size_para,262144),1.0)
    #small=mx.nd.full((size_para,262144),0.01)
    #push loop
    tic0=time.time()               #start time at beginning
    print ("after full", tic0)
    sf=10

    for iter in range(1000):
        #print("handle before push pull",ini_array.handle)
        tic=time.time()
        print ('!!!!!!!!!!!!!!!!!!send start ,key', tic,key[0])
        kv.push(key[0],grad)
        #mx.nd.waitall()
        #kv.pull(key[0],ini_array)
        print ('!!!!!!!!!!!!!!!!!!1wait all')
        mx.nd.waitall()
        #print("handle after push pull",ini_array.handle)
        print("!!!!!!!!!!!!!!!!!!step %d, average push time is %f"%(iter,time.time()-tic))

    print("average push time is %f"%((time.time()-tic0)/1000))
    print("finished")

def main():
    args = parse_args()
    print('Called with argument:', args)
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    kvtest(args)

if __name__ == '__main__':
    main()
