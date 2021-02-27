import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  N,_,H,W=x.shape #input size
  F,_,HH,WW=w.shape #filter size
  
  Hhat = 1+(H + 2*pad - HH) // stride
  What = 1+(W + 2*pad - WW) // stride

  out = np.zeros((N,F,Hhat,What))

  # only want to pad W and H not N and C
  pad_width = ((0,0),(0,0),(pad,pad),(pad,pad))
  xpad = np.pad(x,pad_width,'constant')
  for n in np.arange(N):
    for i in np.arange(Hhat):
      for j in np.arange(What):
        # start of original layer relating to i and j
        h_i = i*stride 
        w_j = j*stride
        # defining the x segment the filter is on
        x_seg = xpad[n,:,h_i:h_i+HH,w_j:w_j+WW]
        out[n,:,i,j] = np.sum(x_seg*w, axis=(1,2,3))+b

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  # preallocations
  db = np.zeros(b.shape)
  dw = np.zeros(w.shape)
  dx_pad = np.zeros(xpad.shape)

  _,_,H,W=x.shape
  _,_,HH,WW=w.shape
  
  Hhat = 1+(H + 2*pad - HH) // stride
  What = 1+(W + 2*pad - WW) // stride

  # Rotation Implementation --Unsure how to incorporate dilation
  # db = np.sum(dout,axis=(0,2,3))
  # dw = np.convolve(xpad,dout)
  # y_pad= ((0,0), (0,0), (pad,pad), (pad,pad))
  # # rotating 180deg
  # w_rot = np.rot90(w,2,axes = (2,3))
  # dx = np.convolve(np.pad(dout,y_pad,'constant'),w_rot)

  db = np.sum(dout,axis=(0,2,3))
  for n in np.arange(N):
    for f in np.arange(F):
      for i in np.arange(Hhat):
        for j in np.arange(What):
          # start of original layer relating to i and j
          h_i = i*stride 
          w_j = j*stride
          # defining the x segment the filter is on
          x_seg = xpad[n,:,h_i:h_i+HH,w_j:w_j+WW]
          dw[f,:,:,:] += dout[n,f,i,j]*x_seg
          dx_pad[n,:,h_i:h_i+HH,w_j:w_j+WW] += dout[n,f,i,j]*w[f,:,:,:] 
  # unpadding
  dx = dx_pad[:,:,pad:H+pad,pad:W+pad]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  N,C,H,W = x.shape
  Hp, Wp, stride = [pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']]

  Hhat = 1 + (H - Hp)// stride 
  What = 1 + (W - Wp)// stride

  out = np.zeros((N,C,Hhat,What))

  for n in np.arange(N):
    for c in np.arange(C):
      for i in np.arange(Hhat):
        for j in np.arange(What):
          # start of original layer relating to i and j
          h_i = i*stride 
          w_j = j*stride
          # defining the x segment the filter is on
          x_seg = x[n,c,h_i:h_i+Hp,w_j:w_j+Wp]
          out[n,c,i,j] = np.max(x_seg)


  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  N,C,H,W = x.shape

  Hhat = 1 + (H - pool_height)// stride 
  What = 1 + (W - pool_width)// stride

  dx = np.zeros(x.shape)

  for n in np.arange(N):
    for c in np.arange(C):
      for i in np.arange(Hhat):
        for j in np.arange(What):
          # start of original layer relating to i and j
          h_i = i*stride 
          w_j = j*stride
          # defining the x segment the filter is on
          x_seg = x[n,c,h_i:h_i+pool_height,w_j:w_j+pool_width]
          # creating indicatior function if x_a > x_b then df/dx_a = 1 OW 0
          ind = (x_seg == np.max(x_seg)) 

          dx[n,c,h_i:h_i+pool_height,w_j:w_j+pool_width] += ind*dout[n,c,i,j]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = x.shape
  x = x.reshape((N*H*W,C))
  
  out,cache = batchnorm_forward(x, gamma, beta, bn_param)
  
  # out has shape (N*H*W,C)
  out = out.T #(C,N*H*W)
  out = out.reshape(C,N,H,W)
  out = out.swapaxes(0,1)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = dout.shape
  dout = dout.swapaxes(0,1)
  dout = dout.reshape((C,N*H*W))
  dout = dout.T #(N*H*W,C)

  dx, dgamma, dbeta = batchnorm_backward(dout, cache)
  dx = dx.reshape((N, C, H, W))
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta