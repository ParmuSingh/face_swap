'''
epochs completed: 150 + 50 + 150 + 50 + 100

'''

import tensorflow as tf
import numpy as np
# from cv2 import imread, imshow
from PIL import Image

base_path1 = "E:/workspace_py/datasets/Siraj images/processed/"
base_path2 = "E:/workspace_py/datasets/Parmu images/"

# base_path = 'E:/Datasets!!/face-swap/face-swap/data/trump'
data = []
parmu_data = []

for i in range(503):
	path = base_path1 + str(i) + '.png'
	# path = r'E:/Datasets!!/face-swap/face-swap/data/trump/ \d*.png'
	# print(path)
	try:
		img = Image.open(path)
		img = np.asarray(img)
		img = np.resize(img, [112*112])
		data.append(img)
	except:
		print("err")
		continue
	path = base_path2 + "Parmu" + str(i) + '.png'

	try:
		img = Image.open(path)
		img = np.asarray(img)
		img = np.resize(img, [112*112])
		parmu_data.append(img)
	except:
		continue

# data = np.reshape(data, [-1, 112*112])
print(len(data))
print(len(parmu_data))
# model

n_epochs = 10
n_examples = 475
batch_size = 1

data_ph = tf.placeholder('float', [None, 112*112], name = 'data_ph')
output_ph = tf.placeholder('float', [None, 112*112], name = 'output_ph')
learning_rate = tf.placeholder('float', [], name = 'learning_rate_ph')

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# def de_conv2d(x, W, output_shape):
# 	return tf.nn.conv2d_transpose(x, W, strides=[1, 1, 1, 1], padding  = 'SAME', output_shape=output_shape)

def maxpool2d(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# def unpool(value, name='unpool'): # https://github.com/tensorflow/tensorflow/issues/2169
#     """N-dimensional version of the unpooling operation from
#     https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

#     :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
#     :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
#     """
#     with tf.name_scope(name) as scope:
#         sh = value.get_shape().as_list()
#         dim = len(sh[1:-1])
#         out = (tf.reshape(value, [-1] + sh[-dim:]))
#         for i in range(dim, 0, -1):
#             out = tf.concat(i, [out, tf.zeros_like(out)])
#         out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
#         out = tf.reshape(out, out_size, name=scope)
#     return out
weights_encoder = {
				'w_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])), # [filter_h, filter_w, in_channels, n_filters]
				'w_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])), # [filter_h, filter_w, in_channels, n_filters]
				'w_conv3': tf.Variable(tf.random_normal([5, 5, 64, 128])),
				'w_fc': tf.Variable(tf.random_normal([25088, 500])),
				'w_out': tf.Variable(tf.random_normal([500, 200]))
				}

biases_encoder = {
				'b_conv1': tf.Variable(tf.random_normal([32])),
				'b_conv2': tf.Variable(tf.random_normal([64])),
				'b_conv3': tf.Variable(tf.random_normal([128])),
				'b_fc': tf.Variable(tf.random_normal([500])),
				'b_out': tf.Variable(tf.random_normal([200]))
				}


weights_decoder_A = {
	'hl1': tf.Variable(tf.random_normal([200, 300])),
	'hl2': tf.Variable(tf.random_normal([300, 500])),
	'hl3': tf.Variable(tf.random_normal([500, 700])), # middle layer
	'ol': tf.Variable(tf.random_normal([700, 112*112])),
	# 'hl5': tf.Variable(tf.random_normal([250, 500])),
	# 'ol': tf.Variable(tf.random_normal([500, 256*256]))
	}

biases_decoder_A = {
	'hl1': tf.Variable(tf.random_normal([300])),
	'hl2': tf.Variable(tf.random_normal([500])),
	'hl3': tf.Variable(tf.random_normal([700])),
	'ol': tf.Variable(tf.random_normal([112*112])),
	# 'hl5': tf.Variable(tf.random_normal([500])),
	# 'ol': tf.Variable(tf.random_normal([256*256]))
	}

weights_decoder_B = {
		'hl1': tf.Variable(tf.random_normal([200, 300])),
		'hl2': tf.Variable(tf.random_normal([300, 500])),
		'hl3': tf.Variable(tf.random_normal([500, 700])), # middle layer
		'ol': tf.Variable(tf.random_normal([700, 112*112])),
		# 'hl5': tf.Variable(tf.random_normal([250, 500])),
		# 'ol': tf.Variable(tf.random_normal([500, 256*256]))
	}

biases_decoder_B = {
		'hl1': tf.Variable(tf.random_normal([300])),
		'hl2': tf.Variable(tf.random_normal([500])),
		'hl3': tf.Variable(tf.random_normal([700])),
		'ol': tf.Variable(tf.random_normal([112*112])),
		# 'hl5': tf.Variable(tf.random_normal([500])),
		# 'ol': tf.Variable(tf.random_normal([256*256]))
	}
	
def encoder(x):
	
	global weights_encoder
	global biases_encoder
	x = tf.reshape(x, [-1, 112, 112, 1])

	conv1 = maxpool2d(conv2d(x, weights_encoder['w_conv1']))
	conv2 = maxpool2d(conv2d(conv1, weights_encoder['w_conv2']))
	conv3 = maxpool2d(conv2d(conv2, weights_encoder['w_conv3']))
	fc = tf.reshape(conv3, [-1, 25088])
	fc = tf.nn.relu(tf.add(tf.matmul(fc, weights_encoder['w_fc']), biases_encoder['b_fc']))
	ol = tf.nn.relu(tf.add(tf.matmul(fc, weights_encoder['w_out']), biases_encoder['b_out']))

	return ol

def decoder_A(x): # Someone make them de-convolution layers.
	
	global weights_decoder_A
	global weights_decoder_A

	hl1 = tf.nn.relu(tf.add(tf.matmul(x, weights_decoder_A['hl1']), biases_decoder_B['hl1']), name = 'hl1')
	hl2 = tf.nn.relu(tf.add(tf.matmul(hl1, weights_decoder_A['hl2']), biases_decoder_A['hl2']), name = 'hl2')
	hl3 = tf.nn.relu(tf.add(tf.matmul(hl2, weights_decoder_A['hl3']), biases_decoder_A['hl3']), name = 'hl3')
	ol = tf.nn.relu(tf.add(tf.matmul(hl3, weights_decoder_A['ol']), biases_decoder_A['ol']), name = 'ol')

	return ol
	# hl5 = tf.nn.relu(tf.add(tf.matmul(hl4, weights['hl5']), biases['hl5']), name = 'hl5')
	# ol = tf.nn.relu(tf.add(tf.matmul(hl1, weights['ol']), biases['ol']), name = 'ol')

def decoder_B(x): # Someone make them de-convolution layers.
	
	global weights_decoder_B
	global biases_decoder_B

	hl1 = tf.nn.relu(tf.add(tf.matmul(x, weights_decoder_B['hl1']), biases_decoder_B['hl1']), name = 'hl1')
	hl2 = tf.nn.relu(tf.add(tf.matmul(hl1, weights_decoder_B['hl2']), biases_decoder_B['hl2']), name = 'hl2')
	hl3 = tf.nn.relu(tf.add(tf.matmul(hl2, weights_decoder_B['hl3']), biases_decoder_B['hl3']), name = 'hl3')
	ol = tf.nn.relu(tf.add(tf.matmul(hl3, weights_decoder_B['ol']), biases_decoder_B['ol']), name = 'ol')

	return ol


# def train_autoencoder():
lossA = tf.reduce_mean((decoder_A(encoder(data_ph)) - output_ph)**2, name = 'loss')
lossB = tf.reduce_mean((decoder_B(encoder(data_ph)) - output_ph)**2, name = 'loss')
trainA = tf.train.AdamOptimizer(learning_rate).minimize(lossA)
trainB = tf.train.AdamOptimizer(learning_rate).minimize(lossB)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# saver = tf.train.import_meta_graph("E:/workspace_py/saved_models/face_swap/autoencoder/autoencoder-1.ckpt.meta")
# saver.restore(sess, tf.train.latest_checkpoint('E:/workspace_py/saved_models/face_swap/autoencoder/'))

errA = 999999 # infinity
errB = 999999
for epoch in range(n_epochs):
	ptr = 0
	for iteration in range(int(n_examples/batch_size)):
		epoch_x = data[ptr : ptr + batch_size]
		epoch_y = parmu_data[ptr : ptr + batch_size]
		ptr += batch_size
		# if errA < 9250:
		# 	_, errA = sess.run([trainA, lossA], feed_dict={data_ph: epoch_x, output_ph: epoch_x, learning_rate: 0.001})
		# else:
		_, errA = sess.run([trainA, lossA], feed_dict={data_ph: epoch_x, output_ph: epoch_x, learning_rate: 0.005})
		# if errB < 1687:
		# 	_, errB = sess.run([trainB, lossB], feed_dict={data_ph: epoch_x, output_ph: epoch_y, learning_rate: 0.01})
		# else:
		_, errB = sess.run([trainB, lossB], feed_dict={data_ph: epoch_x, output_ph: epoch_y, learning_rate: 0.001})

	print("Loss @ epoch ", str(epoch), " = ", errA, " and ", errB)
	# if (epoch + 1) % 50 == 0:
		# save_path = saver.save(sess, "E:/workspace_py/saved_models/face_swap/autoencoder/autoencoder-1.ckpt")

# saver = tf.train.Saver()

prediction = sess.run(decoder_A(encoder(data_ph)), feed_dict={data_ph: [parmu_data[0]]})
print("prediction: ", prediction)

import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.imshow(np.reshape(parmu_data[0], [112, 112]))
# plt.show()
plt.subplot(1,2, 2)
plt.imshow(np.reshape(prediction, [112, 112]))
plt.show()

import cv2 as c

face_cascade = c.CascadeClassifier('E:\workspace_py\OpenCV Cascades\haarcascades\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = c.CascadeClassifier('E:\workspace_py\OpenCV Cascades\haarcascades\haarcascades\haarcascade_eye.xml')


cap = c.VideoCapture(0)
# fourcc = c.VideoWriter_fourcc(*'XVID')
out = c.VideoWriter('output.avi',-1, 5.0, (640,480))

while True:
	#Capture frame-by-frame
	ret, frame = cap.read() # ret = frame exist?

	#Our operations on the frame come here
	gray = c.cvtColor(frame, c.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	# face_cascade.load('haarcascade_frontalface_default.xml')
	for (x,y,w,h) in faces:
		img = gray[y:y+h, x:x+w]
		face_size = (h, w)
		img = np.resize(img, [112*112])

		prediction = np.reshape(sess.run(decoder_A(encoder(data_ph)), feed_dict={data_ph: [img]}), [112, 112])
		# print(prediction)
		# plt.imshow(prediction)
		# plt.show()
		print(face_size[0], face_size[1])
		# prediction = tf.images.resize_images(prediction, [face_size[0], face_size[1]])
		prediction = c.resize(prediction, (face_size[0], face_size[1]))
		gray[y:y+h, x:x+w] = prediction



	#desplay the resulting frame
	c.imshow('frame', gray)
	out.write(gray)
	if c.waitKey(1)  == ord('q'):
		break

#When everything done, release the capture
cap.release()
c.destroyAllWindows()
out.release()
sess.close()