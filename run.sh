#!/bin/sh
node=1
dir="./${node}node_test"
hosts="./hosts"
kvstore="dist_sync_device"
#python -u mxnet/tools/launch.py -n ${node} --launcher ssh -H ${hosts} \
python -u mxnet/tools/launch.py -n ${node} --launcher local \
"\
python -u kvstore_speed_test.py \
       --gpu 0,1 \
       --kvstore ${kvstore}"
