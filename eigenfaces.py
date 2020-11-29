
from google.colab import drive
drive.mount('/content/drive')

from PIL import Image
import numpy as np
import glob
import statistics as st

image = Image.open('drive/My Drive/train/subject01.glasses.pgm')
data = np.asarray(ima\ge)
originalShape = data.shape
size = data.shape[0]*data.shape[1]
image_list = []

trainData = np.zeros((size,150))
testData = np.zeros((size,15))
trainImages = glob.glob('drive/My Drive/train/*.pgm')
testImages = glob.glob('drive/My Drive/test/*.pgm')
for i in range(len(trainImages)):
    trainImage=Image.open(trainImages[i])
    image_list.append(trainImage)
    data1 = np.resize(np.asarray(trainImage),(size))
    trainData[:,i] = data1
print(trainData.shape)
test_imageList = []
for i in range(len(testImages)):
    testImage=Image.open(testImages[i])
    test_imageList.append(testImage)
    data2 = np.resize(np.asarray(testImage),(size))
    testData[:,i] = data2
print(testData.shape)

def PCA(data):
    dimen = data.shape
    meanVector = np.resize(np.mean(data,axis=1),(dimen[0],1))
    h = np.ones((1,dimen[1]))
    data = data - np.dot(meanVector,h)
    XTX = (1/dimen[1]-1)*np.dot(data.T,data)
    eigValues, eigVectors = np.linalg.eig(XTX)
    eigVecXXT = np.dot(data,eigVectors)
    return [eigValues,eigVecXXT]


def noOfEigenValues(eigenValues,percentage):
  dimen = eigenValues.shape
  total = np.sum(eigenValues)
  eigSum,i = 0,0
  while((eigSum/total)<percentage):
    eigSum += eigenValues[i]
    i += 1
  return i
vals,vecs = PCA(trainData)
k = noOfEigenValues(vals,0.95)  
indexDsc = list(np.argsort(vals))
vals = vals[indexDsc]
vecs = vecs[:,np.asarray(indexDsc)]
eigenFaces = vecs[:,0:k]

from google.colab.patches import cv2_imshow
dimen1 = eigenFaces.shape
for i in range(dimen1[1]):
  eigFace = np.reshape(eigenFaces[:,i],(originalShape[0],originalShape[1]))
  print("image ",i)
  cv2_imshow(eigFace)

#test face as linear combination
testSample = testData[:,10]
testSample = np.reshape(testSample,(testSample.shape[0],1))
coeffs = np.dot(np.linalg.pinv(eigenFaces),testSample)
coeffs.shape

print("actual image")
cv2_imshow(np.reshape(testSample,(originalShape[0],originalShape[1])))
print("linearly combined image")
newSample = np.dot(eigenFaces,coeffs)
cv2_imshow(np.reshape(newSample,(originalShape[0],originalShape[1])))

coeffMatrix = np.zeros((64,15))
testData.shape
for i in range(testData.shape[1]):
  testSample = testData[:,i]
  testSample = np.reshape(testSample,(testSample.shape[0],1))
  coeffMatrix[:,i] = np.reshape(np.dot(np.linalg.pinv(eigenFaces),testSample),(64))

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
Coeffs_embedded = TSNE(n_components=2).fit_transform(coeffMatrix)
plt.plot(Coeffs_embedded,"or")

