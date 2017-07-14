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

def UnpackVariable(var, num):
  assert len > 0
  if type(var) is list and len(var) == num:
    return var
  else:
    ret = []
    if type(var) is list:
      assert len(var) == 1
      for i in xrange(0, num):
        ret.append(var[0])
    else:
      for i in xrange(0, num):
        ret.append(var)
    return ret

def ConvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, group=1, dilation=1, use_scale=True, lr_mult=1,
    engine=0, conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn',
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias',
    **bn_params):
  if use_bn:
    # parameters for convolution layer with batchnorm.
    kwargs = {
        'param': [dict(lr_mult=lr_mult, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    eps = bn_params.get('eps', 0.001)
    moving_average_fraction = bn_params.get('moving_average_fraction', 0.999)
    use_global_stats = bn_params.get('use_global_stats', False)
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0)],
        'eps': eps,
        'moving_average_fraction': moving_average_fraction,
        }
    bn_lr_mult = lr_mult
    if use_global_stats:
      # only specify if use_global_stats is explicitly provided;
      # otherwise, use_global_stats_ = this->phase_ == TEST;
      bn_kwargs = {
          'param': [
              dict(lr_mult=0, decay_mult=0),
              dict(lr_mult=0, decay_mult=0),
              dict(lr_mult=0, decay_mult=0)],
          'eps': eps,
          'use_global_stats': use_global_stats,
          }
      # not updating scale/bias parameters
      bn_lr_mult = 0
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          'param': [
              dict(lr_mult=bn_lr_mult, decay_mult=0),
              dict(lr_mult=bn_lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=bn_lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    kwargs = {
        'param': [
            dict(lr_mult=lr_mult, decay_mult=1),
            dict(lr_mult=2 * lr_mult, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }

  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  if kernel_h == kernel_w:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output, group=group,
    engine=engine, kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
  else:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output, group=group,
    engine=engine, kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
        stride_h=stride_h, stride_w=stride_w, **kwargs)
  if dilation > 1:
    net.update(conv_name, {'dilation': dilation})
  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  if use_relu:
    relu_name = '{}_relu'.format(conv_name)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)

def DepthwiseConv(net, from_layer, block_name, num_input, num_output, stride, lr_mult=1, use_scale=True, **bn_params):
  out_layer = '{}/{}'.format(block_name, 'dw')
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,num_output=num_input,
              group=num_input, engine=1, kernel_size=3, pad=1, stride=stride, lr_mult=lr_mult,
              use_scale=use_scale, bn_postfix='/bn', scale_postfix='/scale', **bn_params)
  from_layer = out_layer

  out_layer = '{}/{}'.format(block_name, 'sep')
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,num_output=num_output,
              kernel_size=1, pad=0, stride=1, use_scale=use_scale, lr_mult=lr_mult,
              bn_postfix='/bn', scale_postfix='/scale', **bn_params)

def MobileNetBody(net, from_layer, output_pred=False, **bn_params):
  # scale is fixed to 1, thus we ignore it.
  use_scale = True
  lr_mult = 1
  out_layer = 'conv1'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True, num_output=32, kernel_size=3,
    pad=1, stride=2, lr_mult=lr_mult, use_scale=use_scale, bn_postfix='/bn', scale_postfix='/scale', **bn_params)

  DepthwiseConv(net, 'conv1', 'conv2_1', num_input=32, num_output=64, stride=1, lr_mult=lr_mult, use_scale = use_scale, **bn_params)
  DepthwiseConv(net, 'conv2_1/sep', 'conv2_2', num_input=64, num_output=128, stride=2, lr_mult=lr_mult, use_scale = use_scale, **bn_params)
  DepthwiseConv(net, 'conv2_2/sep', 'conv3_1', num_input=128, num_output=128, stride=1, lr_mult=lr_mult, use_scale = use_scale, **bn_params)
  DepthwiseConv(net, 'conv3_1/sep', 'conv3_2', num_input=128, num_output=256, stride=2, lr_mult=lr_mult, use_scale = use_scale, **bn_params)
  DepthwiseConv(net, 'conv3_2/sep', 'conv4_1', num_input=256, num_output=256, stride=1, lr_mult=lr_mult, use_scale = use_scale, **bn_params)
  DepthwiseConv(net, 'conv4_1/sep', 'conv4_2', num_input=256, num_output=512, stride=2, lr_mult=lr_mult, use_scale = use_scale, **bn_params)
  DepthwiseConv(net, 'conv4_2/sep', 'conv5_1', num_input=512, num_output=512, stride=1, lr_mult=lr_mult, use_scale = use_scale, **bn_params)
  DepthwiseConv(net, 'conv5_1/sep', 'conv5_2', num_input=512, num_output=512, stride=1, lr_mult=lr_mult, use_scale = use_scale, **bn_params)
  DepthwiseConv(net, 'conv5_2/sep', 'conv5_3', num_input=512, num_output=512, stride=1, lr_mult=lr_mult, use_scale = use_scale, **bn_params)
  DepthwiseConv(net, 'conv5_3/sep', 'conv5_4', num_input=512, num_output=512, stride=1, lr_mult=lr_mult, use_scale = use_scale, **bn_params)
  DepthwiseConv(net, 'conv5_4/sep', 'conv5_5', num_input=512, num_output=512, stride=1, lr_mult=lr_mult, use_scale = use_scale, **bn_params)
  DepthwiseConv(net, 'conv5_5/sep', 'conv5_6', num_input=512, num_output=1024, stride=2, lr_mult=lr_mult, use_scale = use_scale, **bn_params)
  DepthwiseConv(net, 'conv5_6/sep', 'conv6', num_input=1024, num_output=1024, stride=1, lr_mult=lr_mult, use_scale = use_scale, **bn_params)

  return net