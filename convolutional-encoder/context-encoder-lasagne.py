# -*- coding: utf-8 -*-  
## 使用lasagne构造context-encoder
# 20161128 使用transposedconvolution恢复图像
# 20161128 设计完成网络，删除最后一层tanh神经元
# 20161128 nBottleneck=100, iter=1
# 20161128 DEBUG: 读数据格式为(1, 128, 128), 修改输入输出及网络
# 20161128 修改网络代码，没有改网络结构
# 20161129 使用Conv2DDNNLayer, it works~~~~ DEBUGING~~~
# 20161203 使用图像mask获取原始采样和修复采样，组合到一起
# 20161203 将注释掉的其他形式网络定义函数剪出去了
# 20161208 添加迭代器
# 20161208 添加保存参数代码，每20代保存在opt['model_folder']+opt['modelG_name']+'_'+str(epoch)+'_'+'.npz'中
# 20161208 修改代码，添加discriminator，命名为netD，每20代保存在opt['model_folder']+opt['modelD_name']+'_'+str(epoch)+'_'+'.npz'中
# 20161209 将文件读取写在batch生成器里
# 20161212 修改netG和netD的定义方式，中间添加全连接层
# 20161212 修改网络结构，读取64x64大小图片
# 20161222 修改数据读取方式，读取npz文件，其中images.npz，sampling_images.npz为float类型图片，mask为0-1掩码
# 20161223 提高batch大小
# 20170101 添加梯度误差
# 20170104 去掉allow_input_downcast，避免误差出现Nan，卷积核数据类型声明为float32
# 20170105 暂停使用discriminator
# 20170108 调整wtl2为0.5
# 20170110 将迭代次数改为200
# 20170130 multi-gpus 结构，修复128x128结构图片，修改batchSize大小

import sys
import os
import time

import numpy as np


import multiprocessing
from multiprocessing import Process, Queue


# import pdb

# 全局参数
# batchSize							图片batch大小
# fineSize							图片大小
# nBottleneck						最终feature map的数量
# nef, ngf, ndf						卷积核数量
# nc 								图像通道数
# wtl2								l2误差的权重（如果不用dcgan会弃用）
# wtgrad							梯度误差的权重
# overlapPred						不知道是啥（如果不用dcgan可能会弃用）
# nThreads							torch中读数据的thread数量（弃用）
# niter 							迭代次数
# lr 								学习率
# betal 							不知道是啥，弃用
# modelG_name						netG名称
# modelD_name						netD名称
# save_epoch						经过epoch代保存模型一次
# model_folder						模型保存文件夹
# manualSeed						可能弃用
opt = {
	"batchSize": 10,
	"fineSize": 128,
	"nBottleneck": 1000,
	"nef": 64,
	"ngf": 64,
	"ndf": 64,
	"nc": 1,
	"wtl2": 0.5,
	"wtgrad": 0.05,
	"overlapPred": 0,
	"nThreads": 4,
	"niter": 200,
	"lr": 0.001,
	"betal": 0.5,
	# netG model
	"modelG_name": 'face_netG',
	# netD model
	"modelD_name": 'face_netD',
	"save_epoch": 20,
	"model_folder": './checkpoints/',
	"manualSeed":0
}
	
# data iterative generator，数据batch迭代器，默认batchsize == opt['batchSize'] == 64
# def iterate_minibatches(inputs=real_ctx, targets=real_pic, batchsize=opt['batchSize'], shuffle=False):
# def iterate_minibatches(inputs, targets, batchsize=opt['batchSize'], shuffle=False):
# 	assert len(inputs) == len(targets)
# 	if shuffle:
# 		indices = np.arange(len(inputs))
# 		np.random.shuffle(indices)
# 	for start_idx in range(0, len(inputs)-batchsize+1, batchsize):
# 		if shuffle:
# 			excerpt = indices[start_idx:start_idx+batchsize]
# 		else:
# 			excerpt = slice(start_idx, start_idx+batchsize)
# 		yield inputs[excerpt], targets[excerpt]

def iterate_minibatches(targets, batchsize=opt['batchSize'], shuffle=False):
	# assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(targets))
		np.random.shuffle(indices)
	for start_idx in range(0, len(targets)-batchsize+1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx+batchsize]
		else:
			excerpt = slice(start_idx, start_idx+batchsize)
		yield targets[excerpt]


def train_netG(private_args, Y_train, sampling_masks, inpainted_masks, q_init_params, q_upl_params, q_err):
	# import theano related
	import theano.sandbox.cuda
	theano.sandbox.cuda.use(private_args['gpu'])

	import theano
	import theano.tensor as T
	from theano.tensor.signal.conv import conv2d

	import lasagne
	# from lasagne.layers.dnn import Conv2DDNNLayer
	# from lasagne.layers import InputLayer, BatchNormLayer, TransposedConv2DLayer
	from lasagne.layers import InputLayer, Conv2DLayer, BatchNormLayer, TransposedConv2DLayer
	from lasagne.nonlinearities import LeakyRectify, ScaledTanH, sigmoid

	# 定义generator netG
	def build_netG():
		# InputLayer
		# input is (nc)*128*128
		network = InputLayer(shape=(None, opt['nc'], opt['fineSize'], opt['fineSize']))
		network = Conv2DLayer(network, num_filters=opt['nef'], filter_size=(4, 4),
			stride=(2, 2), pad=1, nonlinearity=LeakyRectify(0.2))
		# state size: (nef)*64*64
		network = BatchNormLayer(network)
		network = Conv2DLayer(network, num_filters=opt['nef'], filter_size=(4, 4),
			stride=(2, 2), pad=1, nonlinearity=LeakyRectify(0.2))
		# state size: (nef)*32*32
		network = BatchNormLayer(network)
		network = Conv2DLayer(network, num_filters=2*opt['nef'], filter_size=(4, 4),
			stride=(2, 2), pad=1, nonlinearity=LeakyRectify(0.2))
		# state size: (nef*2)*16*16
		network = BatchNormLayer(network)
		network = Conv2DLayer(network, num_filters=4*opt['nef'], filter_size=(4, 4),
			stride=(2, 2), pad=1, nonlinearity=LeakyRectify(0.2))
		# state size: (nef*4)*8*8
		network = BatchNormLayer(network)
		network = Conv2DLayer(network, num_filters=8*opt['nef'], filter_size=(4, 4),
			stride=(2, 2), pad=1, nonlinearity=LeakyRectify(0.2))
		# state size: (nef*8)*4*4
		network = BatchNormLayer(network)
		network = Conv2DLayer(network, num_filters=opt['nBottleneck'], filter_size=(4, 4),
			nonlinearity=LeakyRectify(0.2))
		# state size: (nBottleneck)*1*1
		network = BatchNormLayer(network)
		network = TransposedConv2DLayer(network, num_filters=8*opt['ngf'], filter_size=(4, 4),
			stride=(2, 2), crop=0, nonlinearity=LeakyRectify(0))
		# state size: (ngf*8)*4*4
		network = BatchNormLayer(network)
		network = TransposedConv2DLayer(network, num_filters=4*opt['ngf'], filter_size=(4, 4), 
			stride=(2, 2), crop=1, nonlinearity=LeakyRectify(0))
		# state size: (ngf*4)*8*8
		network = BatchNormLayer(network)
		network = TransposedConv2DLayer(network, num_filters=2*opt['ngf'], filter_size=(4, 4), 
			stride=(2, 2), crop=1, nonlinearity=LeakyRectify(0))
		# state size: (ngf*2)*16*16
		network = BatchNormLayer(network)
		network = TransposedConv2DLayer(network, num_filters=opt['ngf'], filter_size=(4, 4), 
			stride=(2, 2), crop=1, nonlinearity=LeakyRectify(0))
		# state size: (ngf)*32*32
		network = BatchNormLayer(network)
		network = TransposedConv2DLayer(network, num_filters=opt['nc'], filter_size=(4, 4),
			stride=(2, 2), crop=1, nonlinearity=LeakyRectify(0))
		# state size: (nc)*64*64
		network = BatchNormLayer(network)
		network = TransposedConv2DLayer(network, num_filters=opt['nc'], filter_size=(4, 4),
			stride=(2, 2), crop=1, nonlinearity=LeakyRectify(0))
		# state size: (nc)*128*128
		return network

	# 训练符号定义
	# 定义输入变量
	# input_var 输入
	input_var = T.tensor4()
	# target_var 目标输出
	target_var = T.tensor4()
	# sampling_images 采样图片
	sampling_images = T.tensor4()
	# inpainted_images 修复图片
	inpainted_images = T.tensor4()

	# tnsr_s_mask 采样mask
	tnsr_s_mask = T.tensor4()
	# tnsr_i_mask 修复mask（将原先存在的部分去除掉）
	tnsr_i_mask = T.tensor4()

	# tnsr_inpainted 修复结果
	tnsr_inpainted = T.tensor4()

	# grad_x_kern the gradient kernel in the coordinate of x
	grad_x_k = T.matrix('x_kernel')
	# grad_y_kern the gradient kernel in the coordinate of y
	grad_y_k = T.matrix('y_kernel')

	netG = build_netG()

	# 20170101 add the gradient error

	# 1. netG generates
	predG = lasagne.layers.get_output(netG, input_var)
	tnsr_inpainted = predG*tnsr_i_mask+input_var*tnsr_s_mask

	# 20170103 add the gradient error
	images_grad_x, images_x_updates = theano.scan(lambda x, k: abs(conv2d(x, k)), 
		sequences=target_var, non_sequences=grad_x_k)
	images_grad_y, images_y_updates = theano.scan(lambda x, k: abs(conv2d(x, k)),
		sequences=target_var, non_sequences=grad_y_k)
	images_grad = images_grad_x+images_grad_y

	sampling_images_grad_x, sampling_images_x_updates = theano.scan(lambda x, k: abs(conv2d(x, k)),
		sequences=tnsr_inpainted, non_sequences=grad_x_k)
	sampling_images_grad_y, sampling_images_y_updates = theano.scan(lambda x, k: abs(conv2d(x, k)),
		sequences=tnsr_inpainted, non_sequences=grad_y_k)
	sampling_images_grad = sampling_images_grad_x+sampling_images_grad_y

	G_loss2 = lasagne.objectives.squared_error(tnsr_inpainted, target_var).mean()
	G_loss_grad = lasagne.objectives.squared_error(images_grad, sampling_images_grad).mean()
	G_loss = opt['wtl2']*G_loss2+(1-opt['wtl2'])*G_loss_grad

	params_G = lasagne.layers.get_all_params(netG, trainable=True)
	grad_G = T.grad(G_loss, params_G)
	updates_G = lasagne.updates.adam(grad_G, params_G, learning_rate=opt['lr'])

	train_fn_G = theano.function([input_var, target_var, tnsr_s_mask, tnsr_i_mask, 
		grad_x_k, grad_y_k], G_loss, updates=updates_G)

	# define the f_inpaint function to have images inpainted
	inpainted_part = lasagne.layers.get_output(netG, sampling_images)
	inpainted_images = sampling_images*tnsr_s_mask+inpainted_part*tnsr_i_mask
	f_inpaint = theano.function([sampling_images, tnsr_s_mask, tnsr_i_mask], inpainted_images)

	# init parmas of network
	param_values = q_init_params.get()
	lasagne.layers.set_all_param_values(netG, param_values)


	grad_x_kern = np.array([[-1., 1.], 
						    [ 0., 0.]],
						    dtype=np.float32)

	grad_y_kern = np.array([[-1., 0.], 
						 	[ 1., 0.]],
						 	dtype=np.float32)

	train_err_G = 0
	train_batches = 0

	for batch in iterate_minibatches(Y_train):
		target_pic = np.asarray(batch, dtype=np.float32)/255.0
		input_ctx = np.asarray(batch*sampling_masks, dtype=np.float32)/255.0
		train_err_G += train_fn_G(input_ctx, target_pic, sampling_masks, inpainted_masks, 
			grad_x_kern, grad_y_kern)
		train_batches += 1

	param_values = lasagne.layers.get_all_param_values(netG)

	q_upl_params.put(param_values)
	q_err.put(train_err_G/train_batches)


def main():
	# initial private_args
	p1_args = {}
	p1_args['gpu'] = 'cpu'

	p2_args = {}
	p2_args['gpu'] = 'cpu'

	global q_init_params1, q_upl_params1, q_err1
	global q_init_params2, q_upl_params2, q_err2

	if opt['manualSeed'] == 0:
		opt['manualSeed'] = np.random.randint(1, 10000)
	print "Seed:", opt['manualSeed']

	print 'opt:', opt

	# # 读取采样数据和真实数据
	real_pic = np.load('images.npz')['arr_0']
	print 'real_pic.shape:', real_pic.shape
	print 'real_pic.dtype:', real_pic.dtype

	# 读取采样mask和修复mask
	sampling_mask = np.load('mask.npz')['sampling_mask']
	inpainted_mask = np.load('mask.npz')['inpainted_mask']

	# 将其扩展为4维np.array，注意batch_size
	sampling_masks = np.ndarray((opt['batchSize'], 1, 
		sampling_mask.shape[0], sampling_mask.shape[1]), dtype=np.float32)
	inpainted_masks = np.ndarray((opt['batchSize'], 1,
		inpainted_mask.shape[0], inpainted_mask.shape[1]), dtype=np.float32)
	for i in xrange(opt['batchSize']):
		sampling_masks[i] = sampling_mask.reshape(-1, sampling_mask.shape[0], sampling_mask.shape[1])
		inpainted_masks[i] = inpainted_mask.reshape(-1, inpainted_mask.shape[0], inpainted_mask.shape[1])

	# 最终sampling_masks和inpainted_masks为采样和修复掩码4darray
	Y_train1, Y_train2 = real_pic[:-len(real_pic)/2], real_pic[-len(real_pic)/2:]

	# initial the param_values for the q_init_params1 and q_init_params2
	with np.load('init_params.npz') as f:
		param_values = [f['arr_%d'%i] for i in range(len(f.files))]
	q_init_params1.put(param_values)
	q_init_params2.put(param_values)

	for epoch in xrange(opt['niter']):
		start_time = time.time()
		p1 = Process(target=train_netG, args=(p1_args, Y_train1, sampling_masks, inpainted_masks, q_init_params1, q_upl_params1, q_err1, ))
		p2 = Process(target=train_netG, args=(p2_args, Y_train2, sampling_masks, inpainted_masks, q_init_params2, q_upl_params2, q_err2, ))
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

		Y_train1, Y_train2 = Y_train2, Y_train1

		train_err_G = (q_err1.get()+q_err2.get())/2.0

		# if (epoch+1)%opt['save_epoch'] == 0:
		# 	np.savez_compressed(opt['model_folder']+opt['modelG_name']+'_'+str(epoch)+'_'+'.npz',
  #           	*lasagne.layers.get_all_param_values(netG))
		# 	print 'save success!'
		if (epoch+1)%opt['save_epoch'] == 0:
			np.savez_compressed(opt['model_folder']+opt['modelG_name']+'_'+str(epoch)+'_'+'.npz',
				*param_values)

		print 'Time: {}\nEpoch {} of {} tooks {:.3f}s, average train loss G {}\n'.format(
		time.asctime(), epoch+1, opt['niter'], time.time()-start_time, train_err_G)

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    q_init_params1, q_upl_params1, q_err1 = manager.Queue(), manager.Queue(), manager.Queue()
    q_init_params2, q_upl_params2, q_err2 = manager.Queue(), manager.Queue(), manager.Queue()
    main()