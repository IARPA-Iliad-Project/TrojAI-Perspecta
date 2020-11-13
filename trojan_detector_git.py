# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import skimage.io
import random
import math
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
#from scipy.spatial.distance import pdist
import torch
import torch.nn as nn
from torch.autograd import Variable
import warnings 
warnings.filterwarnings("ignore")


def trojan_detector(model_filepath, result_filepath, scratch_dirpath, examples_dirpath, example_img_format='png'):

	print('model_filepath = {}'.format(model_filepath))
	print('result_filepath = {}'.format(result_filepath))
	print('scratch_dirpath = {}'.format(scratch_dirpath))
	print('examples_dirpath = {}'.format(examples_dirpath))

	model = torch.load(model_filepath, map_location=torch.device('cuda'))
	exemplars = dict()
	# Inference the example images in data
	fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith(example_img_format)]
	random.shuffle(fns)
	cloud = np.empty((196608,0), float)
	labels = []
	num_perturb = 50
	for fn in range(len(fns)):
		ex = fns[fn]
		label = [get_class(ex)]
		if label[0] in exemplars:
			if exemplars[label[0]] >= 1:
				continue
		if label[0] not in exemplars: exemplars[label[0]] = 1
		# read the image (using skimage)
		img = skimage.io.imread(ex)
		r = img[:, :, 0]
		g = img[:, :, 1]
		b = img[:, :, 2]
		img = np.stack((r, g, b), axis=2)
		# perform tensor formatting and normalization explicitly
		# convert to CHW dimension ordering
		img = np.transpose(img, (2, 0, 1))
		# convert to NCHW dimension ordering
		img = np.expand_dims(img, 0)
		# normalize the image
		img = img - np.min(img)
		img = img / np.max(img)
		# Find noise that causes misclassification
		for noise in np.arange(0.1,3,0.1):
			errors = 0
			input_label = torch.tensor(label)
			batch_data = torch.FloatTensor(img)
			epsilon = np.random.uniform(-1*noise,noise,batch_data.shape)
			perturbation = batch_data + (torch.tensor(epsilon))
			perturbation = torch.FloatTensor(perturbation.float())
			perturbation = perturbation.cuda()
			logits = model(perturbation)
			sf = nn.Softmax(dim=1)
			result = sf(logits)
			# We want a 99% confident misclassification
			if np.argmax(result.detach()) != input_label and torch.max(result) >= 0.99:
				errors += 1
			if errors > 1:
				break
		# Build clouds of perturbations
		for j in range(num_perturb):
			model.zero_grad()
			input_label = torch.tensor(label)
			batch_data = torch.FloatTensor(img)
			epsilon = np.random.uniform(-1*noise,noise,batch_data.shape)
			perturbation = batch_data + (torch.tensor(epsilon))
			#perturbation = torch.clamp(perturbation, min=0, max=1)
			perturbation = torch.FloatTensor(perturbation.float())
			perturbation.requires_grad=True
			perturbation = perturbation.cuda()
			logits = model(perturbation)
			# Get gradient of logit with respect to input
			gradients = torch.autograd.grad(outputs=logits[0], inputs=perturbation,
	  		grad_outputs=torch.ones(logits[0].size()).to("cuda"), only_inputs=True)[0]
			adv_dir = gradients
			adv_dir = torch.flatten(adv_dir)
			adv_dir = adv_dir.to('cpu')
			adv_dir = adv_dir.detach().numpy()
			adv_dir = np.expand_dims(adv_dir, 1)
			cloud = np.append(cloud, adv_dir, axis=1)
			labels += label

	labelsn = np.array(labels)	
	key_pairs = set()
	class_scores = dict()
	# Look at 2 classes at a time
	for key in exemplars:
		for key2 in exemplars:
			if key != key2:
				if (key,key2) in key_pairs: continue
				else:
					key_pairs.add((key,key2))
					key_pairs.add((key2,key))
				class1 = np.argwhere(labelsn==key)
				class2 = np.argwhere(labelsn==key2)

				features = np.transpose(cloud)[np.union1d(class1, class2),:]
				#features = normalize(features, axis=1)
				pca = PCA(n_components=3)
				features = pca.fit_transform(scale(features))
				features = features.reshape((num_perturb*2,3))
				#features = normalize(features, axis=1)
				labels2class = labelsn[np.union1d(class1, class2)]
				X = features

				kmeans = KMeans(init='k-means++', n_clusters=2)
				labels2 = kmeans.fit_predict(features)
				try: score1 = silhouette_score(features, labels2)
				except: score1 = 0
				if key not in class_scores:
					class_scores[key] = [score1]
				else:
					class_scores[key].append(score1)
				if key2 not in class_scores:
					class_scores[key2] = [score1]
				else:
					class_scores[key2].append(score1)
	max_ss = 0
	for class_key in class_scores:
		avg_score = np.mean(class_scores[class_key])
		ss = 1-avg_score
		if ss > max_ss:
			max_ss = ss
	if max_ss >= 0.5:
		troj_prob = 0.75
	else:
		troj_prob = 0.25

	print('Trojan Probability: {}'.format(troj_prob))

	with open(result_filepath, 'w') as fh:
		fh.write("{}".format(troj_prob))

def get_class(image):
	i = image.index("class")
	if image[i+7:i+8].isnumeric():
		label = image[i+6:i+8]
	else:
		label = image[i+6:i+7]
	return int(label)

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
	parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model.pt')
	parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output')
	parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
	parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./example')


	args = parser.parse_args()
	trojan_detector(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)
