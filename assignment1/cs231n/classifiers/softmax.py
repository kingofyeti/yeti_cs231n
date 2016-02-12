import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  D = W.shape[0] # dimension ---> 4073
  C = W.shape[1] # class -> 10
  N = X.shape[0] # number -> 500

  # X --> N * D 500 * 4073
  # W --> D * C 4073 * 10

  for n in xrange(N):
      scores = X[n,:].dot(W)

      # shift : set all negative as the maximum one as 0
      scores -= np.max(scores)
      exp_scores = np.exp(scores)
      norm_scores = exp_scores / np.sum(exp_scores)

      # negative log : x >> 0, loss >> Inf
      #                x >> 1, loss >> 0
      loss += np.log(np.sum(exp_scores)) - scores[y[n]]

      for c in xrange(C):
        #  if norm_scores[c] > norm_scores[y[n]]:
        #      dW[:,y[n]] -= X[n,:].T
        #      dW[:,c] += X[n,:].T
        dW[:,c] += (norm_scores[c] - (c == y[n])) * X[n,:].T

  loss /= N
  dW /= N

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

