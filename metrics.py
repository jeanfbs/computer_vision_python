import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from medpy.metric import binary as mbin
import sys, csv, os

dataset = "JAI"

indexesName = ["mgrvi", "gli", "mpri", "rgvbi", "ExG", "veg"]

for indexName in indexesName:

    processedDir = './{}/cortadas/{}/'.format(dataset, indexName)
    
    metricsDir = './{}/metrics/'.format(dataset, indexName)
    if not os.path.exists(metricsDir):
        os.makedirs(metricsDir)

    outputMetrics = metricsDir + '{}.csv'.format(indexName)
    filenames = glob.glob("./{}/cortadas/labels/*.tif".format(dataset))
    
    with open(outputMetrics, 'w') as csvfile:
        
        fieldNames = ['Image', 'Dice', 'Jaccard', 'Precision', 'Recall', 'Sensitivity', 'Specificity']    
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(fieldNames)

        for maskFileName in filenames:

            fileName = maskFileName.replace('./{}/cortadas/labels/'.format(dataset), '')
            mask = cv2.imread(maskFileName, cv2.IMREAD_COLOR)
            processed = cv2.imread(processedDir + fileName, cv2.IMREAD_COLOR)

            dc = str(mbin.dc(mask, processed)).replace(".", ",")
            jc = str(mbin.jc(mask, processed)).replace(".", ",")

            precision = str(mbin.precision(processed, mask)).replace(".", ",")
            recall = str(mbin.recall(processed, mask)).replace(".", ",")
            
            sensitivity = str(mbin.sensitivity(processed, mask)).replace(".", ",")
            specificity = str(mbin.specificity(processed, mask)).replace(".", ",")

            writer.writerow([fileName, dc, jc, precision, recall, sensitivity, specificity])
