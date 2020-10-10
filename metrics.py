import glob
import cv2
from matplotlib import pyplot as plt
from medpy.metric import binary as mbin
import sys, csv, os

dataset = "GP"

indexesName = ["mgrvi", "gli", "mpri", "rgvbi", "ExG", "veg"]

for indexName in indexesName:

    processedDir = './{}/cortadas/{}/'.format(dataset, indexName)
    
    metricsDir = './{}/metrics/'.format(dataset, indexName)
    if not os.path.exists(metricsDir):
        os.makedirs(metricsDir)

    outputMetrics = metricsDir + '{}.csv'.format(indexName)
    filenames = glob.glob("./{}/cortadas/labels/*.tif".format(dataset))
    images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in filenames]    
    
    with open(outputMetrics, 'w') as csvfile:
        
        fieldNames = ['Image', 'Dice', 'Jaccard', 'Precision', 'Recall', 'Sensitivity', 'Specificity']    
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(fieldNames)
        
        img_number = 0
        for mask in images:
            
            fileName = filenames[img_number].replace('./{}/cortadas/labels/'.format(dataset), '')
            
            processed = cv2.imread(processedDir + fileName, cv2.IMREAD_COLOR)
            
            dc = str(mbin.dc(processed, mask)).replace(".", ",")
            jc = str(mbin.jc(processed, mask)).replace(".", ",")

            precision = str(mbin.precision(processed, mask)).replace(".", ",")
            recall = str(mbin.recall(processed, mask)).replace(".", ",")
            
            sensitivity = str(mbin.sensitivity(processed, mask)).replace(".", ",")
            specificity = str(mbin.specificity(processed, mask)).replace(".", ",")

            writer.writerow([fileName, dc, jc, precision, recall, sensitivity, specificity])
            img_number = img_number + 1
