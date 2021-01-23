# Neural-Networks-from-scratch

## Contents 

* Project's main topics 
  * A [simple example CNN](#A)
  * The [Net](#The) object
* [Layers](#Layers) 
  * [Linear](#Linear)
  * [Conv2D](#Conv2D)
  * [MaxPool2D](#MaxPool2D)
  * [BatchNorm2D](#BatchNorm2D)
  * [Flatten](#Flatten)
* [Losses](#Losses)
  * [CrossEntropyLoss](#CrossEntropyLoss)
  * [MeanSquareLoss](#MeanSquareLoss)
* [Activations](#Activations)


# Project's main topics 
## A simple example CNN

Its required argument is

* --dataset: path to the dataset,
while the optional arguments are

* --epochs: number of epochs,
* --batch_size: size of the training batch,
* --lr: learning rate.


## The Net object
To define a neural network, the nn.net.Net object can be used. Its parameters are

* layers: a list of layers from nn.layers, for example [Linear(2, 4), ReLU(), Linear(4, 2)],
* loss: a loss function from nn.losses, for example CrossEntropyLoss or MeanSquareLoss. 

If you would like to train the model with data X and label y, you should perform the forward pass, during which local gradients are calculated,
calculate the loss,perform the backward pass, where global gradients with respect to the variables and layer parameters are calculated,
update the weights.

In code, this looks like the following:
* ` out = net(X) `
* `loss = net.loss(out, y) `
* `net.backward()`
* `net.update_weights(lr)`

# Layers

## Linear

A simple fully connected layer. 

Parameters:

* `in_dim`: integer, dimensions of the input.
* `out_dim`: integer, dimensions of the output.

Usage:
* input: ` numpy.ndarray  ` of shape `(N, in_dim)`.
* output: `numpy.ndarray` of shape `(N, out_dim)`.

## Conv2D

2D convolutional layer. Parameters:

* ` in_channels `: integer, number of channels in the input image.
* ` out_channels `: integer, number of filters to be learned.
* ` kernel_size `: integer or tuple, the size of the filter to be learned. Defaults to 3.
* `  stride`: integer, stride of the convolution. Defaults to 1.
* `padding `: integer, number of zeros to be added to each edge of the images. Defaults to 0.

Usage:

* input: ` numpy.ndarray ` of shape (N, C_in, H_in, W_in).
* output: ` numpy.ndarray ` of shape (N, C_out, H_out, W_out).

## MaxPool2D

2D max pooling layer. 

Parameters:

* `kernel_size` : integer or tuple, size of the pooling window. Defaults to 2.

Usage:

* input: `numpy.ndarray ` of shape `(N, C, H, W)`.
* output: `numpy.ndarray ` of shape` (N, C, H//KH, W//KW) `with kernel size `(KH, KW)`.

## BatchNorm2D

2D batch normalization layer. Parameters:

* ` n_channels `: integer, number of channels.
* ` epsilon `: epsilon parameter for BatchNorm, defaults to 1e-5.

Usage:

* input: `numpy.ndarray>` of shape `(N, C, H, W)`.
* output: `numpy.ndarray>` of shape `(N, C, H, W)`.


## Flatten
A simple layer which flattens the outputs of a 2D layer for images.

Usage:

* input : `numpy.ndarray` of shape `(N, C, H, W)`.
* output  : `numpy.ndarray ` of shape` (N, C*H*W)`.


# Losses

## CrossEntropyLoss

Cross-entropy loss. Usage:

*  input: ` numpy.ndarray` of shape `(N, D)` containing the class scores for each element in the batch.
*  output : float.


## MeanSquareLoss

Mean square loss. Usage:

* input : `numpy.ndarray ` of shape `(N, D)`.
* output : `numpy.ndarray` of shape `(N, D)`.

# Activations
The activation layers for the network can be found in nn.activations. They are functions, applying the specified activation function elementwisely on a numpy.ndarray. 
Currently, the following activation functions are implemented:

* ReLU
* Leaky ReLU
* Sigmoid
