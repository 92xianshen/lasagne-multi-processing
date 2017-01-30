# -*- coding: utf-8 -*-

# log for code
# 20170126 model parallelism, half of minibatch is deployed in each model, and average the params
# 20170126 model parallelism, half of the whole dataset is deployed in each model, and average the params

import numpy as np
import time
import os
import sys

import multiprocessing
from multiprocessing import Process, Queue

import theano
import theano.tensor as T

import lasagne

def build_mlp():
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28))

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out

def build_cnn():
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28))
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs)-batchsize+1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx+batchsize]
        else:
            excerpt = slice(start_idx, start_idx+batchsize)
        yield inputs[excerpt], targets[excerpt]

def data_loader(training_set, training_labels):
    with np.load(training_set) as f:
        miniX = f['arr_0']
    with np.load(training_labels) as f:
        miniX_labels = f['arr_0']
    return miniX, miniX_labels

# def train_network(q_init_params, q_upl_params, q_err, minibatch, minibatch_label):
#     network = build_mlp()

#     input_var = T.tensor4()
#     target_var = T.ivector()

#     prediction = lasagne.layers.get_output(network, input_var)
#     loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
#     loss = loss.mean()

#     params = lasagne.layers.get_all_params(network, trainable=True)
#     updates = lasagne.updates.nesterov_momentum(
#         loss, params, learning_rate=0.01, momentum=0.9)

#     train_fn = theano.function(inputs=[input_var, target_var],
#         outputs=loss, updates=updates)

#     param_values = q_init_params.get()
#     lasagne.layers.set_all_param_values(network, param_values)

#     train_err = train_fn(minibatch, minibatch_label)

#     param_values = lasagne.layers.get_all_param_values(network)

#     q_upl_params.put(param_values)
#     q_err.put(train_err)



# def main():
#     # initial params
#     training_set = 'X_train.npz'
#     training_labels = 'y_train.npz'
#     num_epochs = 100
#     batchsize = 500
#     model_folder = 'checkpoints/'
#     model_name = 'multi-mlp-network'

#     # define global queues
#     global q_init_params1, q_upl_params1, q_err1
#     global q_init_params2, q_upl_params2, q_err2

#     # load dataset and its labels
#     miniX, miniX_labels = data_loader(training_set, training_labels)

#     # build mlp
#     network = build_mlp()

#     q_init_params1.put(lasagne.layers.get_all_param_values(network))
#     q_init_params2.put(lasagne.layers.get_all_param_values(network))

#     for epoch in xrange(num_epochs):
#         train_err = 0
#         train_batches = 0
#         start_time = time.time()
#         for batch in iterate_minibatches(miniX, miniX_labels, batchsize, shuffle=False):
#             inputs, targets = batch
#             inputs1, inputs2 = inputs[:-batchsize/2], inputs[-batchsize/2:]
#             targets1, targets2 = targets[:-batchsize/2], targets[-batchsize/2:]
#             # print 'inputs1.shape, inputs1.dtype:', inputs1.shape, inputs1.dtype
#             # print 'inputs2.shape, inputs2.dtype:', inputs2.shape, inputs2.dtype
#             # print 'targets1.shape, targets1.dtype:', targets1.shape, targets1.dtype
#             # print 'targets2.shape, targets2.dtype:', targets2.shape, targets2.dtype
#             p1 = Process(target=train_network, args=(q_init_params1, q_upl_params1, q_err1, inputs1, targets1, ))
#             p2 = Process(target=train_network, args=(q_init_params2, q_upl_params2, q_err2, inputs2, targets2, ))
#             p1.start()
#             p2.start()
#             p1.join()
#             p2.join()
#             params_num = 0
#             while True:
#                 if not q_upl_params1.empty():
#                     param_values1 = q_upl_params1.get()
#                     params_num += 1
#                 if not q_upl_params2.empty():
#                     param_values2 = q_upl_params2.get()
#                     params_num += 1
#                 if params_num == 2:
#                     break

#             param_values = [(param_values1[i]+param_values2[i])/2.0 for i in xrange(len(param_values1))]
            
#             q_init_params1.put(param_values)
#             q_init_params2.put(param_values)

#             train_err += (q_err1.get()+q_err2.get())/2
#             train_batches += 1

#         if (epoch+1)%10 == 0:
#             np.savez_compressed(model_folder+model_name+'_'+str(epoch), *param_values)

#         print 'Epoch {} of {} tooks {:.3f}s'.format(
#             epoch+1, num_epochs, time.time()-start_time)
#         print 'training loss:\t\t{:.6f}'.format(train_err/train_batches)


def train_network(X_train, y_train, batchsize, q_init_params, q_upl_params, q_err):
    # network = build_mlp()
    network = build_cnn()

    input_var = T.tensor4()
    target_var = T.ivector()

    prediction = lasagne.layers.get_output(network, input_var)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

    train_fn = theano.function(inputs=[input_var, target_var],
        outputs=loss, updates=updates)

    param_values = q_init_params.get()
    lasagne.layers.set_all_param_values(network, param_values)

    train_err = 0
    train_batches = 0
    for batch in iterate_minibatches(X_train, y_train, batchsize, shuffle=False):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1

    param_values = lasagne.layers.get_all_param_values(network)

    q_upl_params.put(param_values)
    q_err.put(train_err/train_batches)

def main():
    # initial params
    training_set = 'X_train.npz'
    training_labels = 'y_train.npz'
    val_set = 'X_val.npz'
    val_labels = 'y_val.npz'
    num_epochs = 100
    batchsize = 500
    model_folder = 'checkpoints/'
    # model_name = 'multi-mlp-network'
    model_name = 'multi-cnn-network'
    # build mlp
    cnn = build_cnn()

    # validation
    val_inputs_var = T.tensor4()
    val_targets_var = T.ivector()
    val_prediction = lasagne.layers.get_output(cnn, val_inputs_var)
    val_loss = lasagne.objectives.categorical_crossentropy(val_prediction, val_targets_var)

    val_loss = val_loss.mean()
    val_acc = T.mean(T.eq(T.argmax(val_prediction, axis=1), val_targets_var), 
        dtype=theano.config.floatX)
    val_fn = theano.function([val_inputs_var, val_targets_var], [val_loss, val_acc])

    # define global queues
    global q_init_params1, q_upl_params1, q_err1
    global q_init_params2, q_upl_params2, q_err2

    # load dataset and its labels
    X_train, y_train = data_loader(training_set, training_labels)
    X_val, y_val = data_loader(val_set, val_labels)

    X_train1, X_train2 = X_train[:-len(X_train)/2], X_train[-len(X_train)/2:]
    y_train1, y_train2 = y_train[:-len(y_train)/2], y_train[-len(y_train)/2:]

    q_init_params1.put(lasagne.layers.get_all_param_values(cnn))
    q_init_params2.put(lasagne.layers.get_all_param_values(cnn))

    for epoch in xrange(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        p1 = Process(target=train_network, args=(X_train1, y_train1, batchsize, q_init_params1, q_upl_params1, q_err1, ))
        p2 = Process(target=train_network, args=(X_train2, y_train2, batchsize, q_init_params2, q_upl_params2, q_err2, ))
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        params_num = 0
        while True:
            if not q_upl_params1.empty():
                param_values1 = q_upl_params1.get()
                params_num += 1
            if not q_upl_params2.empty():
                param_values2 = q_upl_params2.get()
                params_num += 1
            if params_num == 2:
                break

        param_values = [(param_values1[i]+param_values2[i])/2.0 for i in xrange(len(param_values1))]
        
        q_init_params1.put(param_values)
        q_init_params2.put(param_values)

        X_train1, X_train2 = X_train2, X_train1
        y_train1, y_train2 = y_train2, y_train1

        train_err += (q_err1.get()+q_err2.get())/2
        train_batches += 1

        if (epoch+1)%10 == 0:
            np.savez_compressed(model_folder+model_name+'_'+str(epoch), *param_values)

        print 'Epoch {} of {} tooks {:.3f}s'.format(
            epoch+1, num_epochs, time.time()-start_time)
        print 'training loss:\t\t{:.6f}'.format(train_err/train_batches)

        lasagne.layers.set_all_param_values(cnn, param_values)

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batchsize, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        print 'validation loss:\t\t{:.6f}'.format(val_err/val_batches)
        print 'validation accuracy:\t\t{:.2f} %'.format(val_acc/val_batches*100)




if __name__ == '__main__':
    manager = multiprocessing.Manager()
    q_init_params1, q_upl_params1, q_err1 = manager.Queue(), manager.Queue(), manager.Queue()
    q_init_params2, q_upl_params2, q_err2 = manager.Queue(), manager.Queue(), manager.Queue()
    main()

