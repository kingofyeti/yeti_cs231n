import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    
    self.params['W1'] = np.random.normal(0,weight_scale,input_dim*hidden_dim).reshape(input_dim,hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = np.random.normal(0,weight_scale,hidden_dim*num_classes).reshape(hidden_dim,num_classes)
    self.params['b2'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################

    W1 = self.params['W1']
    b1 = self.params['b1']
    W2 = self.params['W2']
    b2 = self.params['b2']

    l1_out, l1_cache = affine_forward(x=X, w=W1, b=b1)
    l2_out, l2_cache = relu_forward(x=l1_out) 
    l3_out, l3_cache = affine_forward(x=l2_out, w=W2, b=b2)

    scores = l3_out

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################

    l4_out,dl3_out_x = softmax_loss(x=l3_out, y=y) 

    # Loss with regularization
    loss = l4_out+0.5*self.reg*(np.sum(W1*W1)+np.sum(W2*W2))

    dl2_out_x, dl2_out_w, dl2_out_b = affine_backward(dout=dl3_out_x, cache=l3_cache)
    dl1_out_x = relu_backward(dout=dl2_out_x, cache=l2_cache)
    dl0_x, dl0_w, dl0_b = affine_backward(dout=dl1_out_x, cache=l1_cache)
    
    grads['W1'] = dl0_w + self.reg*W1
    grads['b1'] = dl0_b
    grads['W2'] = dl2_out_w + self.reg*W2
    grads['b2'] = dl2_out_b

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    
    pre_layer_dim = input_dim

    for i,cur_hidden_dim in enumerate(hidden_dims):
      self.params['W'+str(i+1)] = np.random.normal(0,weight_scale,pre_layer_dim*cur_hidden_dim).reshape(pre_layer_dim,cur_hidden_dim)
      self.params['b'+str(i+1)] = np.zeros(cur_hidden_dim)
      pre_layer_dim = cur_hidden_dim
      if self.use_batchnorm:
        self.params['gamma'+str(i+1)] = 1
        self.params['beta'+str(i+1)] = 0
    
      self.params['W'+str(self.num_layers)] = np.random.normal(0,weight_scale,pre_layer_dim*num_classes).reshape(pre_layer_dim,num_classes)
      self.params['b'+str(self.num_layers)] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave dkfferently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None

    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################

    self.params['l0_relu_out'] = X

    for i in range(1,self.num_layers):
      cur_W = self.params['W'+str(i)]
      cur_b = self.params['b'+str(i)]
      self.params['l'+str(i)+'_affine_out'], self.params['l'+str(i)+'_affine_cache'] = affine_forward(x=self.params['l'+str(i-1)+'_relu_out'], w=cur_W, b=cur_b)
      self.params['l'+str(i)+'_relu_out'], self.params['l'+str(i)+'_relu_cache'] = relu_forward(x=self.params['l'+str(i)+'_affine_out']) 

    last_affine_layer = self.num_layers

    self.params['l'+str(last_affine_layer)+'_affine_out'], self.params['l'+str(last_affine_layer)+'_affine_cache'] = affine_forward(x=self.params['l'+str(last_affine_layer-1)+'_relu_out'], w=self.params['W'+str(last_affine_layer)], b=self.params['b'+str(last_affine_layer)])

    scores = self.params['l'+str(last_affine_layer)+'_affine_out']
    self.params['softmax_out'],self.params['dl'+str(last_affine_layer)+'_affine_out'] = softmax_loss(x=self.params['l'+str(last_affine_layer)+'_affine_out'], y=y) 
 
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    pass

    # Loss with regularization
    loss = self.params['softmax_out']

    sum_W = 0;

    for i in range(1,self.num_layers+1):
        cur_W = self.params['W'+str(i)]
        sum_W += np.sum(cur_W * cur_W)
    loss += 0.5 * self.reg * sum_W

    
    self.params['dl'+str(last_affine_layer-1)+'_relu_out'],self.params['dl'+str(last_affine_layer-1)+'_relu_out_w'],self.params['dl'+str(last_affine_layer-1)+'_relu_out_b'] = affine_backward(dout = self.params['dl'+str(last_affine_layer)+'_affine_out'],cache = self.params['l'+str(last_affine_layer)+'_affine_cache'])

    for i in range(last_affine_layer-1,0,-1):
      self.params['dl'+str(i)+'_affine_out'] = relu_backward(dout=self.params['dl'+str(i)+'_relu_out'],cache=self.params['l'+str(i)+'_relu_cache'])
      self.params['dl'+str(i-1)+'_relu_out'],self.params['dl'+str(i-1)+'_relu_out_w'],self.params['dl'+str(i-1)+'_relu_out_b'] = affine_backward(dout = self.params['dl'+str(i)+'_affine_out'],cache = self.params['l'+str(i)+'_affine_cache'])
    
    for i in range(1,last_affine_layer+1):
      grads['W'+str(i)] = self.params['dl'+str(i-1)+'_relu_out_w'] + self.reg * self.params['W'+str(i)]
      grads['b'+str(i)] = self.params['dl'+str(i-1)+'_relu_out_b']

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
