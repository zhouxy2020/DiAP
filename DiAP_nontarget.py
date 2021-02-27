import matplotlib.pyplot as plt
import tensorflow as tf
import math
from matplotlib import pylab as P

import os
import numpy as np
import pickle
from io import StringIO
import PIL.Image
import time
import glob
import random

import keras
from keras import applications
from keras import backend as K
import numpy as np
from scipy import misc
import cv2
import imagenet_stubs
from imagenet_stubs.imagenet_2012_labels import label_to_name, name_to_label

TARGET_LABEL = name_to_label('toaster') # Try "banana", "Pembroke, Pembroke Welsh corgi"
PATCH_SHAPE = (299, 299, 3)
BATCH_SIZE = 10

# Ensemble of models
NAME_TO_MODEL = {
    'xception': applications.xception.Xception,
    'vgg16': applications.vgg16.VGG16,
    'vgg19': applications.vgg19.VGG19,
    'resnet50': applications.resnet50.ResNet50,
    'inceptionv3': applications.inception_v3.InceptionV3,
}

MODEL_NAMES = [ 'inceptionv3',#'resnet50','vgg16','xception','vgg19',
                ]

# Data augmentation
# Empirically found that training with a very wide scale range works well
# as a default
SCALE_MIN = 0.1
SCALE_MAX = 1.0
MAX_ROTATION = 45

def _convert(im):
	return ((im + 1) * 127.5).astype(np.uint8)

def show(im):
	plt.axis('off')
	plt.imshow(_convert(im), interpolation="nearest")
	plt.show()

def load_image(image_path):
	im = PIL.Image.open(image_path)
	im = im.resize((299, 299), PIL.Image.ANTIALIAS)
	if image_path.endswith('.png'):
		ch = 4
	else:
		ch = 3
	try:
		im = np.array(im.getdata()).reshape(im.size[0], im.size[1], ch)[:, :, :3]
	except:
		im = np.array(im.getdata()).reshape(im.size[0], im.size[1], 1)
		temp = np.zeros((299, 299, 3))
		temp[:, :, 0] = im[:, :, 0]
		temp[:, :, 1] = im[:, :, 0]
		temp[:, :, 2] = im[:, :, 0]
		im = np.copy(temp)
	return im / 127.5 - 1

class StubImageLoader():
	"""An image loader that uses just a few ImageNet-like images.
	In the actual paper, we used real ImageNet images, but we can't include them
	here because of licensing issues.
	"""
	def __init__(self):
		self.images = []
		self.toaster_image = None

		for image_path in imagenet_stubs.get_image_paths():
			im = load_image(image_path)

			if image_path.endswith('toaster.jpg'):
				self.toaster_image = im
			else:
				self.images.append(im)
	def get_images(self):
		# return random.sample(self.images, BATCH_SIZE)
		return self.images

image_loader = StubImageLoader()

def _transform_vector(width, x_shift, y_shift, im_scale, rot_in_degrees):
	"""
	 If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1],
	 then it maps the output point (x, y) to a transformed input point
	 (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k),
	 where k = c0 x + c1 y + 1.
	 The transforms are inverted compared to the transform mapping input points to output points.
	"""
	rot = float(rot_in_degrees) / 90. * (math.pi / 2)
	# Standard rotation matrix
	# (use negative rot because tf.contrib.image.transform will do the inverse)
	rot_matrix = np.array(
		[[math.cos(-rot), -math.sin(-rot)],
		 [math.sin(-rot), math.cos(-rot)]]
	)
	inv_scale = 1. / im_scale
	xform_matrix = rot_matrix * inv_scale
	a0, a1 = xform_matrix[0]
	b0, b1 = xform_matrix[1]
	# At this point, the image will have been rotated around the top left corner,
	# rather than around the center of the image.
	# To fix this, we will see where the center of the image got sent by our transform,
	# and then undo that as part of the translation we apply.
	x_origin = float(width) / 2
	y_origin = float(width) / 2

	x_origin_shifted, y_origin_shifted = np.matmul(
		xform_matrix,
		np.array([x_origin, y_origin]),
	)
	x_origin_delta = x_origin - x_origin_shifted
	y_origin_delta = y_origin - y_origin_shifted
	# Combine our desired shifts with the rotation-induced undesirable shift
	a2 = x_origin_delta - (x_shift / (2 * im_scale))
	b2 = y_origin_delta - (y_shift / (2 * im_scale))
	# Return these values in the order that tf.contrib.image.transform expects
	return np.array([a0, a1, a2, b0, b1, b2, 0, 0]).astype(np.float32)

# @title class ModelState()

def _circle_mask(shape, sharpness=40):
	"""Return a circular mask of a given shape"""
	assert shape[0] == shape[1], "circle_mask received a bad shape: " + shape

	diameter = shape[0]
	x = np.linspace(-1, 1, diameter)
	y = np.linspace(-1, 1, diameter)
	xx, yy = np.meshgrid(x, y, sparse=True)
	z = (xx ** 2 + yy ** 2) ** sharpness

	mask = 1 - np.clip(z, -1, 1)
	mask = np.expand_dims(mask, axis=2)
	mask = np.broadcast_to(mask, shape).astype(np.float32)
	return mask

def _gen_target_ys():
	label = TARGET_LABEL
	y_one_hot = np.zeros(1000)
	y_one_hot[label] = 1.0
	y_one_hot = np.tile(y_one_hot, (BATCH_SIZE, 1))
	return y_one_hot

TARGET_ONEHOT = _gen_target_ys()

class ModelContainer():
	"""Encapsulates an Imagenet model, and methods for interacting with it."""

	def __init__(self, model_name, verbose=True, peace_mask=None, peace_mask_overlay=0.0):
		# Peace Mask: None, "Forward", "Backward"
		self.model_name = model_name
		self.graph = tf.Graph()
		self.sess = tf.compat.v1.Session(graph=self.graph)
		self.peace_mask = peace_mask
		self.patch_shape = PATCH_SHAPE
		self._peace_mask_overlay = peace_mask_overlay
		self.load_model(verbose=verbose)

	def patch(self, new_patch=None):
		"""Retrieve or set the adversarial patch.

		new_patch: The new patch to set, or None to get current patch.

		Returns: Itself if it set a new patch, or the current patch."""
		if new_patch is None:
			return self._run(self._clipped_patch)

		self._run(self._assign_patch, {self._patch_placeholder: new_patch})
		return self

	def reset_patch(self):
		"""Reset the adversarial patch to all zeros."""
		# self.patch(np.zeros(self.patch_shape))
		self.patch(np.random.uniform(size=self.patch_shape))


	def train_step(self, images=None, target_ys=None, learning_rate=5.0, scale=(0.1, 1.0),
	               par_inp = [1. ,0.], par_loss=[1., 0.],
	               dropout=None):
		"""Train the model for one step.

		Args:
		  images: A batch of images to train on, it loads one if not present.
		  target_ys: Onehot target vector, defaults to TARGET_ONEHOT
		  learning_rate: Learning rate for this train step.
		  scale: Either a scalar value for the exact scale, or a (min, max) tuple for the scale range.

		Returns: Loss on the target ys."""
		if images is None:
			images = np.zeros((BATCH_SIZE,299,299,3))
		if target_ys is None:
			target_ys = TARGET_ONEHOT

		feed_dict = {self._image_input: images,
		             self._target_ys: target_ys,
		             self._learning_rate: learning_rate,
		             self._par_inp: par_inp,
		             self._par_loss: par_loss}

		loss, sal_adv, _ = self._run([self._loss, self._sal_adv, self._train_op], feed_dict, scale=scale, dropout=dropout)
		return loss, sal_adv

	def inference_batch(self, images=None, target_ys=None, scale=None, par_inp=None):
		"""Report loss and label probabilities, and patched images for a batch.
		Args:
		  images: A batch of images to train on, it loads if not present.
		  target_ys: The target_ys for loss calculation, TARGET_ONEHOT if not present."""
		if images is None:
			images = image_loader.get_images()
		if target_ys is None:
			target_ys = TARGET_ONEHOT
		if par_inp is None:
			par_inp = [1., 0.]
		feed_dict = {self._image_input: images, self._target_ys: target_ys,self._par_inp: par_inp,}
		loss_per_example, ps, ims, output_layer = self._run([self._loss_per_example,
		                            self._probabilities, self._inp,
		                                                     # self._patched_input,
		                                                     self.ouput_layer ],
		                                      feed_dict, scale=scale)
		return loss_per_example, ps, ims, output_layer

	def load_model(self, verbose=True):
		model = NAME_TO_MODEL[self.model_name]
		if self.model_name in ['xception', 'inceptionv3', 'mobilenet']:
			keras_mode = False
		else:
			keras_mode = True
		patch = None
		self._make_model_and_ops(model, keras_mode, patch, verbose)

	def _run(self, target, feed_dict=None, scale=None, dropout=None):
		K.set_session(self.sess)
		if feed_dict is None:
			feed_dict = {}
		feed_dict[self.learning_phase] = False
		if scale is not None:
			if isinstance(scale, (tuple, list)):
				scale_min, scale_max = scale
			else:
				scale_min, scale_max = (scale, scale)
			feed_dict[self.scale_min] = scale_min
			feed_dict[self.scale_max] = scale_max
		if dropout is not None:
			feed_dict[self.dropout] = dropout
		return self.sess.run(target, feed_dict=feed_dict)

	def l2_all(self, network):
		loss = 0
		j = 0
		for i in network:
			# if i in layers:
			if 'conv'  in i.name:
				loss += tf.math.log(tf.nn.l2_loss(tf.abs(i)))
				j += 1
			elif 'res'  in i.name:
				loss += tf.math.log(tf.nn.l2_loss(tf.abs(i)))
				j += 1
		return loss

	def _make_model_and_ops(self, M, keras_mode, patch_val, verbose):
		start = time.time()
		K.set_session(self.sess)
		with self.sess.graph.as_default():
			self.learning_phase = K.learning_phase()
			image_shape = (299, 299, 3)
			self._image_input = keras.layers.Input(shape=image_shape)
			self.scale_min = tf.compat.v1.placeholder_with_default(SCALE_MIN, [])
			self.scale_max = tf.compat.v1.placeholder_with_default(SCALE_MAX, [])
			self._scales = tf.random.uniform([BATCH_SIZE], minval=self.scale_min, maxval=self.scale_max)
			image_input = self._image_input
			patch = tf.compat.v1.get_variable("patch", self.patch_shape, dtype=tf.float32, initializer=tf.zeros_initializer)
			self._patch_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=self.patch_shape)
			self._assign_patch = tf.assign(patch, self._patch_placeholder)
			modified_patch = patch
			def clip_to_valid_image(x):
				return tf.clip_by_value(x, clip_value_min=-1., clip_value_max=1.)
			self._clipped_patch = clip_to_valid_image(modified_patch)

			if keras_mode:
				image_input = tf.image.resize(image_input, (224, 224))
				image_shape = (224, 224, 3)
				modified_patch = tf.image.resize(patch, (224, 224))

			self.dropout = tf.compat.v1.placeholder_with_default(1.0, [])
			patch_with_dropout = tf.nn.dropout(modified_patch, keep_prob=self.dropout)
			patched_input = clip_to_valid_image(self._random_overlay(image_input, patch_with_dropout, image_shape))

			def to_keras(x):
				x = (x + 1) * 127.5
				R, G, B = tf.split(x, 3, 3)
				R -= 123.68
				G -= 116.779
				B -= 103.939
				x = tf.concat([B, G, R], 3)
				return x

			# Since this is a return point, we do it before the Keras color shifts
			# (but after the resize, so we can see what is really going on)
			self._patched_input = patched_input

			self._par_inp = tf.compat.v1.placeholder_with_default([1., 0.], shape=[2])
			inp = patched_input * self._par_inp[0] + image_input * self._par_inp[1]
			self._inp = inp
			if keras_mode:
				inp = to_keras(inp)

			# Labels for our attack (e.g. always a toaster)
			self._target_ys = tf.compat.v1.placeholder(tf.float32, shape=(None, 1000))
			model = M(input_tensor=inp, weights='imagenet')

			# Pre-softmax logits of our pretrained model
			self.logits = model.outputs[0].op.inputs[0]
			self._loss_per_example = tf.nn.softmax_cross_entropy_with_logits_v2(
				labels=self._target_ys,
				logits=self.logits
			)
			self._target_loss = tf.reduce_mean(self._loss_per_example)
			self._loss = self._target_loss
			self.ouput_layer = [layer.output for layer in model.layers] #每层的激活值
			self._sal_adv = -self.l2_all(self.ouput_layer)

			# Train our attack by only training on the patch variable
			self._learning_rate = tf.compat.v1.placeholder(tf.float32)
			self._par_loss = tf.compat.v1.placeholder_with_default([0.,1.], shape=[2])
			self._train_op = tf.compat.v1.train.GradientDescentOptimizer(self._learning_rate) \
				.minimize(self._sal_adv* self._par_loss[0] + self._loss* self._par_loss[1], var_list=[patch])

			self._probabilities = model.outputs[0]
			if patch_val is not None:
				self.patch(patch_val)
			else:
				self.reset_patch()

			elapsed = time.time() - start
			if verbose:
				print("Finished loading {}, took {:.0f}s".format(self.model_name, elapsed))

	def _pad_and_tile_patch(self, patch, image_shape):
		# Calculate the exact padding
		# Image shape req'd because it is sometimes 299 sometimes 224

		# padding is the amount of space available on either side of the centered patch
		# WARNING: This has been integer-rounded and could be off by one.
		#          See _pad_and_tile_patch for usage
		return tf.stack([patch] * BATCH_SIZE)

	def _random_overlay(self, imgs, patch, image_shape):
		"""Augment images with random rotation, transformation.

		Image: BATCHx299x299x3
		Patch: 50x50x3

		"""
		# Add padding

		image_mask = _circle_mask(image_shape)
		image_mask = tf.stack([image_mask] * BATCH_SIZE)
		padded_patch = tf.stack([patch] * BATCH_SIZE)

		transform_vecs = []

		def _random_transformation(scale_min, scale_max, width):
			im_scale = np.random.uniform(low=scale_min, high=scale_max)
			padding_after_scaling = (1 - im_scale) * width
			x_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)
			y_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)
			rot = np.random.uniform(-MAX_ROTATION, MAX_ROTATION)
			return _transform_vector(width,
			                         x_shift=x_delta,
			                         y_shift=y_delta,
			                         im_scale=im_scale,
			                         rot_in_degrees=rot)

		for i in range(BATCH_SIZE):
			# Shift and scale the patch for each image in the batch
			random_xform_vector = tf.py_func(_random_transformation, [self.scale_min, self.scale_max, image_shape[0]],
			                                 tf.float32)
			random_xform_vector.set_shape([8])
			transform_vecs.append(random_xform_vector)

		image_mask = tf.contrib.image.transform(image_mask, transform_vecs, "BILINEAR")
		padded_patch = tf.contrib.image.transform(padded_patch, transform_vecs, "BILINEAR")
		inverted_mask = (1 - image_mask)
		return imgs * inverted_mask + padded_patch * image_mask

# @ MetaModel

class MetaModel():
	def __init__(self, verbose=True, peace_mask=None, peace_mask_overlay=0.0):
		self.nc = {m: ModelContainer(m, verbose=verbose, peace_mask=peace_mask, peace_mask_overlay=peace_mask_overlay)
		           for m in MODEL_NAMES}
		self._patch = np.zeros(PATCH_SHAPE)
		self.patch_shape = PATCH_SHAPE

	def patch(self, new_patch=None):
		"""Retrieve or set the adversarial patch.
		new_patch: The new patch to set, or None to get current patch.
		Returns: Itself if it set a new patch, or the current patch."""
		if new_patch is None:
			return self._patch
		self._patch = new_patch
		return self

	def reset_patch(self):
		"""Reset the adversarial patch to all zeros."""
		self.patch(np.zeros(self.patch_shape))

	def train_step(self, model=None, steps=1, images=None, target_ys=None, learning_rate=5.0, par_loss=[1.,0.], scale=None, **kwargs):
		"""Train the model for `steps` steps.

		Args:
		  images: A batch of images to train on, it loads one if not present.
		  target_ys: Onehot target vector, defaults to TARGET_ONEHOT
		  learning_rate: Learning rate for this train step.
		  scale: Either a scalar value for the exact scale, or a (min, max) tuple for the scale range.

		Returns: Loss on the target ys."""

		if model is not None:
			to_train = [self.nc[model]]
		else:
			to_train = self.nc.values()

		losses = []
		sal_adves =[]
		for mc in to_train:
			mc.patch(self.patch())
			for i in range(steps):
				loss, sal_adv = mc.train_step(images, target_ys, learning_rate, scale=scale, par_loss=par_loss, **kwargs)
				losses.append(loss)
				sal_adves.append(sal_adv)
			self.patch(mc.patch())
		return np.mean(losses), np.mean(sal_adves)

	def inference_batch(self, model, images=None, target_ys=None, scale=None):
		"""Report loss and label probabilities, and patched images for a batch.

		Args:
		  images: A batch of images to train on, it loads if not present.
		  target_ys: The target_ys for loss calculation, TARGET_ONEHOT if not present.
		  scale: Either a scalar value for the exact scale, or a (min, max) tuple for the scale range.
		"""

		mc = self.nc[model]
		mc.patch(self.patch())
		return mc.inference_batch(images, target_ys, scale=scale)


print("Creating MetaModel...")
MM = MetaModel()

def show_patch(patch):
	circle = _circle_mask((299, 299, 3))
	show(circle * patch + (1 - circle))

img_list_test = './utils/ilsvrc_1000_test.txt'
def report(model, step=None, show_images=False, verbose=True, scale=(0.1, 1.0)):
	"""Prints a report on how well the model is doing.
	If you want to see multiple samples, pass a positive int to show_images

	Model can be a ModelContainer instance, or a string. If it's a string, we
	lookup that model name in the MultiModel
	"""
	start = time.time()
	# n examples where target was in top 5
	top_5 = 0
	# n examples where target was top 1
	wins = 0
	# n examples where fool success
	fools = 0
	# n examples in total
	img_test = open(img_list_test).readlines()[:100]
	n_batches = int(math.ceil(float(len(img_test)) / BATCH_SIZE))
	total = BATCH_SIZE * n_batches

	loss = 0
	for b in range(n_batches):
		images = []
		for image_path in img_test[b*BATCH_SIZE:(b+1)*BATCH_SIZE]:
			im = load_image(image_path.strip())
			images.append(im)

		if isinstance(model, str):
			loss_per_example, probs, patched_imgs, output_layer = M.inference_batch(model, images=images, scale=scale)
			_, probs_o, imgs, _ = M.inference_batch(model, images=images, scale=scale, par_inp=[0.,1.])
		else:
			loss_per_example, probs, patched_imgs, output_layer = model.inference_batch( images=images,scale=scale)
			_, probs_o, imgs, _ = model.inference_batch( images=images, scale=scale, par_inp=[0., 1.])
		loss += np.mean(loss_per_example)
		# for i in range(BATCH_SIZE):
		for i in range(len(loss_per_example)):
			top_labels = np.argsort(-probs[i])[:5]
			if TARGET_LABEL in top_labels:
				top_5 += 1
				if top_labels[0] == TARGET_LABEL:
					wins += 1
			imgs_labels = np.argmax(probs_o[i])
			if top_labels[0] != imgs_labels:
				fools += 1

	loss = loss / n_batches
	top_5p = int(100 * float(top_5) / total)
	winp = int(100 * float(wins) / total)
	foolp = int(100 * float(fools) / total)

	if step is not None:
		r = 'Step: {} \t'.format(step)
	else:
		r = ''
	r += 'LogLoss: {:.1f} \tWin Rate: {}%\t Top5: {}%\tn: {}'.format(math.log(loss), winp, top_5p, total)
	if verbose:
		print(r)

	if show_images:
		if show_images is True:
			show_images = 1
	elapsed = time.time() - start
	return {'logloss': math.log(loss), 'win': winp, 'top5': top_5p, 'time': elapsed, 'loss': loss, 'fool': foolp,}

def cross_model_report(meta_model, n=100, verbose=True, scale=None):
	results = {}
	print('{:15s}\t Loss\t Win%\t Top5%\t Time'.format('Model Name'))
	out_start = time.time()
	for model_name in MODEL_NAMES:
		model = meta_model.name_to_container[model_name]
		r = report(model, n=n, verbose=False, scale=scale)
		results[model_name] = r
		print('{:15s}\t {:.1f}\t {:.0f}%\t {:.0f}%\t {:.0f}s'.format(model_name, r['loss'], r['win'], r['top5'],
		                                                             r['time']))

	def _avg(name):
		xs = [r[name] for r in results.values()]
		return sum(xs) / len(xs)
	elapsed = time.time() - out_start
	print(
		'{:15s}\t {:.1f}\t {:.0f}%\t {:.0f}%\t {:.0f}s'.format('Average/Total', _avg('loss'), _avg('win'), _avg('top5'),
		                                                       elapsed))
	return results


import pickle
import os.path as osp
from datetime import datetime

def save_obj(obj, file_name):
	serialized = pickle.dumps(obj, protocol=0)
	dest_file = osp.join(DATA_DIR, file_name)
	with open(dest_file, 'wb') as f:
		f.write(serialized)

def load_obj(file_name):
	dest_file = osp.join(DATA_DIR, file_name)
	with open(dest_file, 'rb') as f:
		pkl = f.read()
	return pickle.loads(pkl)

def _latest_snapshot_path(experiment_name):
	"""Return the latest pkl file"""
	return osp.join(DATA_DIR, "%s.latest" % (experiment_name))

def _timestamped_snapshot_path(experiment_name):
	"""Return a timestamped pkl file"""
	timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	return osp.join(DATA_DIR, "%s.%s" % (experiment_name, timestamp))

def save_patch(experiment_name, model):
	"""Save a snapshot for the given experiment"""
	def _serialize_patch(dest_file):
		patch = model.patch()
		serialized = pickle.dumps(patch, protocol=0)  # protocol 0 is printable ASCII
		with open(dest_file + ".pkl", 'w') as f:
			f.write(serialized)
			print("Wrote patch to %s" % dest_file)
		with open(dest_file + ".jpg", 'w') as f:
			PIL.Image.fromarray(_convert(model.patch())).save(f, "JPEG")
	_serialize_patch(_latest_snapshot_path(experiment_name))
	_serialize_patch(_timestamped_snapshot_path(experiment_name))

def load_patch(experiment_name_or_patch_file, model, dontshow=False):
	if experiment_name_or_patch_file.startswith(DATA_DIR):
		patch_file = experiment_name_or_patch_file
	else:
		patch_file = _latest_snapshot_path(experiment_name_or_patch_file)
	with open(patch_file + '.pkl', 'r') as f:
		pkl = f.read()
	patch = pickle.loads(pkl)
	model.patch(patch)
	if not dontshow:
		show_patch(patch)

def get_im(path):
	with open(osp.join(DATA_DIR, path), "r") as f:
		pic = PIL.Image.open(f)
		pic = pic.resize((299, 299), PIL.Image.ANTIALIAS)
		if path.endswith('.png'):
			ch = 4
		else:
			ch = 3
		pic = np.array(pic.getdata()).reshape(pic.size[0], pic.size[1], ch)[:, :, :3]
		pic = pic / 127.5 - 1
	return pic

areas_to_report = list(np.linspace(0.01, 0.10, 5)) + [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

def calculate_win_rates(models):
	start = time.time()
	rows = len(models)
	nontarget = np.zeros((rows, len(areas_to_report)))
	target = np.zeros((rows, len(areas_to_report)))
	for (i, model) in enumerate(models):
		print("Evaluating %s" % model.model_name)
		for (j, a) in enumerate(areas_to_report):
			sc = 2 * math.sqrt(a / math.pi)
			re = report(model, scale=sc, verbose=False)
			target[i, j] = re['win']
			nontarget[i, j] = re['fool']
	print('Calculated wins in {:.0f}s'.format(time.time() - start))
	return nontarget, target

def plot_win_rates(wins, labels, title):
	assert wins.shape[0] == len(labels)
	for (i, l) in enumerate(labels):
		plt.plot([a * 100.0 for a in areas_to_report], wins[i], label=l)
	plt.title(title)
	plt.legend()
	plt.xlabel("Attack as % of image size")
	plt.ylabel("Attack success rate")
	plt.show()

###################
# Local data dir to write files to
DATA_DIR = './content/DiAP_nontarget'

## single attack
model_targets = MODEL_NAMES
STEPS = 800
LR = 8
regular_training_model_to_patch = {}
x = 0
for m in model_targets:
	print("Training %s" % m)
	M = MM.nc[m]
	M.reset_patch()
	for i in range(STEPS):
		x += 1
		loss, sal_adv = M.train_step()
		if i % int(STEPS / 10) == 0:
			print("[%s] loss: %s, l_sal: %s" % (i, loss, sal_adv))
	regular_training_model_to_patch[m] = M.patch()
models=[]
models.append(M)
regular_training_nontarget_rates, regular_training_target_rates = calculate_win_rates(models)
save_obj(regular_training_model_to_patch, "regular_training_model_to_patch")

## ensemble attack
transfer_ensemble_model_names = [ 'resnet50', 'xception',  'inceptionv3' , 'vgg16' ,'vgg19']
print("Beginning ensemble experiment with ensemble: %s " % (
	transfer_ensemble_model_names))
MM.reset_patch()
for i in range(STEPS):
	for mm in transfer_ensemble_model_names:
		loss, sal_adv = MM.train_step(mm, steps=1, learning_rate=LR)
	if i % int(STEPS / 10) == 0:
		print("[%s] loss: %s, l_sal: %s" % (i, loss, sal_adv))

ensemble_patch = MM.patch()
save_obj(ensemble_patch, "ensemble_patch")
ensemble_patch = load_obj("ensemble_patch")

models =[]
for m in transfer_ensemble_model_names:
	M = MM.nc[m]
	M.patch(ensemble_patch)
	models.append(M)

ensemble_attack_win_rates_nontarget, ensemble_attack_win_rates_target  = calculate_win_rates(models)
save_obj(ensemble_attack_win_rates_nontarget, "ensemble_attack_win_rates_nontarget")
save_obj(ensemble_attack_win_rates_target, "ensemble_attack_win_rates_target")

## black attack
transfer_ensemble_model_names = [ 'xception',  'inceptionv3', 'vgg16' ,'vgg19']
transfer_target_model = 'resnet50'

print("Beginning blackbox experiment with ensemble: %s and target %s" % (
	transfer_ensemble_model_names, transfer_target_model))
MM.reset_patch()
for i in range(STEPS):
	for mm in transfer_ensemble_model_names:
		loss, sal_adv = MM.train_step(mm, steps=1, learning_rate=LR)
	if i % int(STEPS / 10) == 0:
		print("[%s] loss: %s, l_sal: %s" % (i, loss, sal_adv))

transfer_patch = MM.patch()
save_obj(transfer_patch, "transfer_patch_resnet50")
m = MM.nc[transfer_target_model]
m.patch(transfer_patch)

transfer_attack_win_rates_nontarget, transfer_attack_win_rates_target  = calculate_win_rates([m])

save_obj(transfer_attack_win_rates_nontarget, "transfer_attack_win_rates_resnet50_nontarget")
save_obj(transfer_attack_win_rates_target, "transfer_attack_win_rates_resnet50_target")
