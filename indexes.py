import glob
import cv2
import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from medpy import metric
import sys, os

#Modified Green Red Vegetation
def mgrvi(G, R, B):
    return (np.power(G, 2) - np.power(R, 2))/(np.power(G, 2) + np.power(R, 2))

def gli(G, R, B):
    return (2*G - R - B) / (2*G + R + B)

def mpri(G, R, B):
    return (G - R) / (G + R)

def rgvbi(G, R, B):
    aux = (np.power(G, 2) + (B * R))
    aux[aux == 0] = 255
    return (G - (B * R)) / aux

def ExG(G, R, B):
    return (2 * G) - R - B

def veg(G, R, B):
    aux = np.power(R, 0.667) * np.power(B, 1 - 0.667)
    aux[aux == 0] = 255
    return G / aux


dataset = "JAI"
indexesName = ["mgrvi", "gli", "mpri", "rgvbi", "ExG", "veg"]

for indexName in indexesName:

    print "running index: " + str(indexName)
    output = './{}/cortadas/{}/'.format(dataset, indexName)
    if not os.path.exists(output):
        os.makedirs(output)

    filenames = glob.glob("./{}/cortadas/images/*.tif".format(dataset))
    
    for origFileName in filenames:

        orig = cv2.imread(origFileName, 1)
        
        fileName = origFileName.replace('./{}/cortadas/images/'.format(dataset), '')
        print 'processing image... :' + str(origFileName)
        image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        image = np.float32(image)

        # Indice
        R = image[:,:, 0]
        G = image[:,:, 1]
        B = image[:,:, 2]

       
        index = globals()[indexName](G, R, B)
        
        # reshape the image to a 2D array of pixels and 3 color values (RGB)
        pixel_values = index.reshape((-1, 1))
        
        # convert to float
        pixel_values = np.float32(pixel_values)

        centers = np.float32([np.min(index), np.max(index)]).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, init=centers, max_iter=300).fit(pixel_values)
        segmented_image = kmeans.labels_.reshape(index.shape)

        binOutput = np.ones((index.shape[0], index.shape[1], 3), dtype=np.uint8)
        newPixel = segmented_image[:,:] * 255
        binOutput[:, :, 0] = newPixel
        binOutput[:, :, 1] = newPixel
        binOutput[:, :, 2] = newPixel

        kernel = np.ones((7,7))
        closing = cv2.morphologyEx(binOutput, cv2.MORPH_OPEN, kernel)
        
        io.imsave(output + fileName, closing)
        
