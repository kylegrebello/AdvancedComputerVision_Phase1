import sys
import getopt
import torch
import torchattacks
import foolbox
import glob
import numpy
import eagerpy as ep
import torchvision.models as models
from foolbox import PyTorchModel, accuracy, samples
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from scipy.io import loadmat
from foolbox.attacks import LinfPGD, L2PGD
from matplotlib import pyplot as plt

def preprocessImage(image):
	returnImage = image.convert('RGB')
	preprocess = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	returnImage = preprocess(returnImage)
	#returnImage = returnImage.unsqueeze(0)
	return returnImage

def loadModel(modelName):
	print("Entering loadModel")
	model = None
	if modelName == 'resnet':
		model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
		model.eval()
	elif modelName == 'vgg':
		model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16_bn', pretrained=True)
		model.eval()
	else:
		print("Invalid Model Name... Please re-run with -m resnet or -m vgg")
		sys.exit()

	print("Exiting loadModel")
	return model

def loadAndPreprocessImages(image_dir):
	print("Entering loadAndPreprocessImages")
	imagenet = datasets.imagenet.ImageNet(image_dir, split='val', download=True)
	preprocessedImages = []
	labels = []

	numberOfImages = len(imagenet.imgs)
	counter = 0
	for image in imagenet.imgs:
		input_image = Image.open(image[0])
		preprocessedImages.append(preprocessImage(input_image))
		labels.append(image[1])
		counter = counter + 1
		if (counter % 1000 == 0):
			print("Image Number: {} Percent Complete: {:.2%}".format(counter, (counter / numberOfImages)))
	print("Exiting loadAndPreprocessImages")
	return preprocessedImages, labels

def runAttackOnImages(model, attackType, images, labels):
	print("Entering runAttackOnImages")
	attack = None
	if attackType == "L2":
		attack = L2PGD()
	elif attackType == "Linf":
		attack = LinfPGD()
	else:
		print("Invalid Attack Type... Please re-run with -a L2 or -a Linf")
		sys.exit()

	fmodel = foolbox.PyTorchModel(model, bounds=(0,1))
	#epsilons = [2,3,4,5,6,7,8,9,10]
	epsilons = [2]
	images = torch.stack(images)
	labels = torch.LongTensor(labels)
	adv_images, advs, success = attack(fmodel, images, labels, epsilons=epsilons)
	print("Exiting runAttackOnImages")

	return adv_images, advs, success

def getAccuracyOfModel(model, images, labels, adversarial):
	print("Entering getAccuracyOfModel")
	incorrect = 0
	numberOfImages = len(images)
	for i in range(numberOfImages):
		if adversarial:
			input_batch = images[i]
		else:
			input_batch = images[i].unsqueeze(0)

		# move the input and model to GPU for speed if available
		if torch.cuda.is_available():
			input_batch = input_batch.to('cuda')
			model.to('cuda')

		with torch.no_grad():
			output = model(input_batch)

		argmax = numpy.argmax(output[0])
		if (argmax != labels[i]):
			incorrect = incorrect + 1

		if (i % 100 == 0 and i != 0):
			print("Image Number: {} Percent Complete: {:.2%}".format(i, (i / numberOfImages)))

	print("Accuracy: {:.2%}".format(1 - (incorrect / numberOfImages)))
	print("Exiting getAccuracyOfModel")

def main(argv):
	modelName = ''
	img_dir = ''
	attack = ''
	try:
		opts, args = getopt.getopt(argv,"h:m:i:a:",["model=","img_dir=","attack="])
	except getopt.GetoptError:
		print('Phase1.py -m <model (resnet | vgg)> -i <image dir> -a <attack (L2 | Linf)>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('Phase1.py -m <model (resnet | vgg)> -i <image dir> -a <attack (L2 | Linf)>')
			sys.exit()
		elif opt in ("-m", "--model"):
			modelName = arg
		elif opt in ("-i", "--img_dir"):
			img_dir = arg
		elif opt in ("-a", "--attack"):
			attack = arg

	model = loadModel(modelName)
	images, labels = loadAndPreprocessImages(img_dir)

	getAccuracyOfModel(model, images, labels, False)

	adv_images, advs, success = runAttackOnImages(model, attack, images, labels)
	images.clear()
	getAccuracyOfModel(model, adv_images, labels, True)

if __name__ == "__main__":
   main(sys.argv[1:])