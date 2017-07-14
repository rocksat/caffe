import os

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

def check_if_exist(path):
    return os.path.exists(path)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def Highway(net, prefix, from_layer, num_output, num_layers,prepare = True):
    assert from_layer in net.keys()
    print(prefix, from_layer, num_output, num_layers)
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='gaussian',std=0.0025),
            'bias_filler': dict(type='xavier')}
#            'weight_filler': dict(type='xavier'),
#            'bias_filler': dict(type='constant', value=0)}
    prev = net[from_layer]
    start = 0
    if prepare:
	print("prepare", "prev", prev)
        prev = L.Convolution(prev, num_output=num_output, pad=1, kernel_size=3, **kwargs)
        net[prefix+"_conv_0"] = prev
        prev = L.ReLU(prev, in_place=True, relu_param={'negative_slope': 0.18})
        name = prefix+"_lrelu_0"
        net[name] = prev
        start = 1



    for i in range(start, num_layers):
        conv = L.Convolution(prev, num_output=num_output, pad=1, kernel_size=3, **kwargs)
        relu = L.ReLU(conv, in_place=True, relu_param={'negative_slope': 0.18})
        prev = L.Eltwise(prev, relu, operation=P.Eltwise.SUM)
        
        net[prefix+"_conv_"+str(i)]=conv
        net[prefix+"_relu_"+str(i)]=relu
        name = prefix+"_hw_"+str(i)
	print(i,"adding", name)
        net[name]=prev

    return name
        
def UHighway(net, from_layer, downscales=[(4, 16), (4,32), (8,64), (8,96), (8,128), (8,128), (16,128), (64,128), (128,128)], upscales=[(32,128),(16,128),(4, 128)]):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    highway_exits = []
    prev = from_layer
    for i, (num_layers, num_output) in enumerate(downscales):
        exit_name = Highway(net, "down"+str(i), prev, num_output, num_layers)
	if i < len(downscales) - 1:
            exit = L.Convolution(net[exit_name], num_output=num_output, pad=1, kernel_size=3, stride = 2, **kwargs)
            prev = "down_exit"+str(i)
            net[prev] = exit
        highway_exits.append(exit_name)

    outputs = [highway_exits[-1]]
    for i, (num_layers, num_output) in enumerate(upscales):
        prev_name = outputs[-1]
        dec = L.Deconvolution(net[prev_name], convolution_param=dict(num_output=num_output, kernel_size=2, stride=2))
        net["up_dec"+str(i)]=dec
        prev = "up_dec"+str(i)
        
        name = "up_sum"+str(i)
        net[name]=L.Eltwise(net[prev], net[highway_exits[-i-2]], operation=P.Eltwise.SUM)
        prev=name

        output_name = Highway(net, "up"+str(i), prev, num_output, num_layers)
        outputs.append(output_name)

    return net,list(reversed(outputs))


