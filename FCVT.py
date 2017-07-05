# -*- coding: utf-8 -*-
"""
@author: ChernHong Lim
"""
import numpy as np
import copy
import os
import matplotlib.pyplot as plt

import cv2

from scipy import stats

from skimage.feature import local_binary_pattern
#from skimage import io

from sklearn.cluster import KMeans
from sklearn.svm import SVC

import FQRC


"""""""""""""""""""""
Image Acquisition
"""""""""""""""""""""

#Read Source
def IA_readSource( sourceDir, display ):   
   image = cv2.imread(sourceDir)       
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   if display:
       show(image)       
   return image


"""""""""""""""""""""
Image Preprocessing
"""""""""""""""""""""
#Convert Gray
def IP_convertGray(image, display):        
    grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    
    if display:
        show(grayImage)        
    return grayImage


#Convert Binary
def IP_convertBinary(image, display):        
    grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (thresh, binaryImage) = cv2.threshold(grayImage, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)    
    if display:
        show(binaryImage)        
    return binaryImage


#Image Resize
def IP_resize(image, sx, sy, display):
    resizedImage = cv2.resize(image,None,fx=sx, fy=sy)
    if display:
        show(resizedImage)  
    return resizedImage 


#Image Filtering
def IP_imageFilt(image, method, kernel, display):
    if method == 'average':        
        filteredImage = cv2.blur(image,kernel)        
    elif method == 'gaussian':        
        filteredImage = cv2.GaussianBlur(image,kernel,0)        
    elif method == 'median':        
        filteredImage = cv2.medianBlur(image,kernel[0])        
    if display:
        show(filteredImage) 
    return filteredImage


#Image Morphological Operation
def IP_imageMorph(image, method, kernelSize, display):
    kernel = np.ones((kernelSize[0],kernelSize[1]),np.uint8)
    if method == 'erosion':        
        morphImage = cv2.erode(image,kernel,iterations = 1)
    elif method == 'dilation':        
        morphImage = cv2.dilate(image,kernel,iterations = 1)
    elif method == 'opening':
        morphImage = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif method == 'closing':
        morphImage = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)        
    if display:
        show(morphImage) 
    return morphImage



"""""""""""""""""""""
Feature Extraction
"""""""""""""""""""""
#Color detection
def FE_colorDetection(image, lowerbound, upperbound, display):
    lower = np.array([lowerbound[0],lowerbound[1],lowerbound[2]]) #lower boundary of RGB value
    upper = np.array([upperbound[0],upperbound[1],upperbound[2]]) #upper boundary of RGB value
    mask_image = cv2.inRange(image, lower, upper)
    colorImage = cv2.bitwise_and(image, image, mask = mask_image)
    if(display):
        show(colorImage)
    return colorImage


#Edge detection
def FE_edgeDetection(image, display):
    edgesImage = cv2.Canny(image,100,200)
    if display:
        show(edgesImage) 
    return edgesImage


#Corner detection
def FE_cornerDetection(image, display):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
#    dst = IP_imageMorph(dst, 'dilation', (10,10), False) #result is dilated for marking the corners, not important
    image[dst>0.01*dst.max()]=[255,0,0] # Threshold for an optimal value, it may vary depending on the image.
    if display:
        show(image) 
    return dst


#Keypoint SIFT, SURF
def FE_keypointDetection(image, method, display):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    if method == 'SIFT':
        points = cv2.SIFT()
    elif method == 'SURF':
        points = cv2.SURF()
    kp, des = points.detectAndCompute(gray,None)
    if display:
        img=cv2.drawKeypoints(gray,kp)
        show(img)
    return kp,des
    

#LBP
def FE_LBPDetection(image, display):
    # settings for LBP
    radius = 3
    n_points = 8 * radius    
    METHOD = 'uniform'
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, METHOD)
    # Calculate the histogram 
    x = stats.itemfreq(lbp.ravel()) 
    # Normalize the histogram 
    hist = x[:, 1]/sum(x[:, 1]) 
    if display:
        print 'imageLBP = [' + " ".join(['%0.4f'%i for i in hist]) + "]"
#        print hist
    return hist
    

#HOG



"""""""""""""""""""""
Feature Representation - Clustering and Quantisation to support Bag of Feature
"""""""""""""""""""""

#Feature clustering
def FE_Clustering(feaMat, clusterNo):    
    cluster = KMeans(n_clusters=clusterNo)  
    cluster.fit(feaMat)        
    cluster_labels = cluster.labels_
    cluster_centers = cluster.cluster_centers_
    cluster_labels_unique = np.unique(cluster_labels)         
    return cluster,cluster_labels,cluster_centers,cluster_labels_unique


#Feature quantisation
def FE_Quantisation(feaMat, cluster):
    cluster_membership = cluster.predict(feaMat)
    descriptor = np.bincount(cluster_membership, minlength=len(cluster.cluster_centers_))
    return descriptor


"""""""""""""""""""""
Classification
"""""""""""""""""""""

def CL_Train(X_train, y_train, method, visualize):
    if method == 'Crisp':
        Classifier = SVC(kernel="linear", C=0.025)
        Classifier.fit(X_train, y_train)
    elif method == 'Fuzzy':
        Classifier = FQRC.CL_FQRC_Train(X_train, y_train, binNum=5, visualize=visualize)
    return Classifier
    

def CL_Predict(X_test, classifier, method, visualize):  
    if method == 'Crisp':        
        Predict = classifier.predict(X_test)  
    elif method == 'Fuzzy':
        Predict = FQRC.CL_FQRC_Predict(X_test, classifier, visualize=visualize)
    return Predict

  
  
"""""""""""""""""""""
Application: Image classification
"""""""""""""""""""""
def Image_Classification(trainingFolder, testingFolder, feature, classification_method):

    """
    Setting
    """
    path = os.getcwd()
    
    #display
    display = False
    visualize = True
    
    
    """
    Training
    """
    pathTraining = path + '\\' + trainingFolder
    dirsTraining = os.listdir(pathTraining)
    noOfClass = len(dirsTraining)
    
    groundTruth = np.arange(0,noOfClass)    
 
    counter2 = 0
    
    if(feature == 'SIFT' or feature == 'SURF'):
        # Keypoint detection for all files in each folder
        for ind in range(0,noOfClass):
            pathTrainingClass = pathTraining + '\\' + dirsTraining[ind]
            dirsTF = os.listdir(pathTrainingClass)    
            listofGT = [ind] * len(dirsTF)
            
            counter1 = 0
            for indFile in range(0,len(dirsTF)):
                image = IA_readSource(pathTrainingClass + '\\' + dirsTF[indFile], display)
                imageKeyPoint = FE_keypointDetection(image, feature, display)
                
                
                if counter1 == 0:
                    imageKeyPoint_perImage = [imageKeyPoint[1]]
                    counter1 = counter1 + 1
                else:
                    imageKeyPoint_perImage.append(imageKeyPoint[1])
            
            if counter2 == 0:
                data = [dirsTF,listofGT,imageKeyPoint_perImage] 
                counter2 = counter2 + 1
            else:
                list.extend(data[0],dirsTF)
                list.extend(data[1],listofGT)
                list.extend(data[2],imageKeyPoint_perImage)
               
                
        # Keypoint clustering
        counter3 = 0
        for item in data[2]:
            if counter3 == 0:
                imageKeyPoint_all = item
                counter3 = counter3 + 1
            else:
                imageKeyPoint_all = np.concatenate((imageKeyPoint_all, item), axis=0)  
                
        cluster = FE_Clustering(imageKeyPoint_all, 5)
        
        
        #Keypoint quantisation
        counter4 = 0;
        for indData in range(0,len(data[1])):
            quantisation = FE_Quantisation(data[2][indData], cluster[0])
        
            if counter4==0:
                desc = copy.copy(quantisation)
                counter4 = counter4 + 1
            else:
                desc = np.vstack((desc,quantisation))
    
    elif(feature == 'LBP'):     
        for ind in range(0,noOfClass):
            pathTrainingClass = pathTraining + '\\' + dirsTraining[ind]
            dirsTF = os.listdir(pathTrainingClass)    
            listofGT = [ind] * len(dirsTF)
            
            counter1 = 0
            for indFile in range(0,len(dirsTF)):
                image = IA_readSource(pathTrainingClass + '\\' + dirsTF[indFile], display)
                imageFea = FE_LBPDetection(image, display)   
                
                if counter1 == 0:
                    imageFea_perImage = [imageFea]
                    counter1 = counter1 + 1
                else:
                    imageFea_perImage.append(imageFea)
                
            if counter2 == 0:
                data = [dirsTF,listofGT,imageFea_perImage] 
                counter2 = counter2 + 1
            else:
                list.extend(data[0],dirsTF)
                list.extend(data[1],listofGT)
                list.extend(data[2],imageFea_perImage)
                
        counter4 = 0
        for indData in range(0,len(data[1])):
            if counter4==0:
                desc = copy.copy(data[2][indData])
                counter4 = counter4 + 1
            else:
                desc = np.vstack((desc,data[2][indData]))
        
    #Classification
    trainDes = desc
    trainGT = np.array(data[1])
    
    classifier = CL_Train(trainDes, trainGT, classification_method, visualize)    
    
    
    """
    Testing
    """
    pathTesting = path + '\\' + testingFolder
    dirsTesting = os.listdir(pathTesting)
    noOfClass = len(dirsTesting)
    
    output_overall = []   
    
    for ind in range(0,noOfClass):
        pathTestingClass = pathTesting + '\\' + dirsTesting[ind]
        dirsTest = os.listdir(pathTestingClass) 
        listofGTTest = [ind] * len(dirsTest)        
        
        counter5 = 0        
        
        for indFile in range(0,len(dirsTest)):
            image = IA_readSource(pathTestingClass + '\\' + dirsTest[indFile], display)  
            
            if(feature == 'SIFT' or feature == 'SURF'):
                imageKeyPoint = FE_keypointDetection(image, feature, display)
                desctest = FE_Quantisation(imageKeyPoint[1], cluster[0])
            elif(feature == 'LBP'):
                desctest = FE_LBPDetection(image, display) 
                
            answer = CL_Predict(desctest, classifier, classification_method, visualize=True)        
            
            if counter5 == 0:
                output = answer
                counter5 = counter5 + 1
            else:
                output = np.vstack((output,answer))#           
            
            fig = plt.figure()
            plt.imshow(image)
            plt.title('Classification Results: ' + str(np.around(answer,decimals=2)))
            plt.axis('off')
                
        output_overall.append(output)
    
    return output_overall
        



    

"""""""""""""""""""""
Utility
"""""""""""""""""""""
#Image visualization
def show(image):
    if(len(image.shape)>2):
       image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#    io.imshow(image)
    









