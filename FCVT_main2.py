# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:49:57 2017

@author: ChernHong
"""

"""
Main testing
"""


#from FCVT import *
import FCVT as fcvt
  


#"""""""""""""""""""""
#Image Acquisition
#"""""""""""""""""""""
##Provide setting
#sourceDir = 'Lenna.png' 
#
##Read image
#image = fcvt.IA_readSource(sourceDir, True)
#
#
#"""""""""""""""""""""
#Image Preprocessing
#"""""""""""""""""""""
#
##Image resize
#imageResize = fcvt.IP_resize(image, 0.5, 0.5, True)
#
##Convert image to gray scale
#imageGray = fcvt.IP_convertGray(image, True)
#imageBinary = fcvt.IP_convertBinary(image, True)
#
##Image Morphological Operation
#imageDigit = fcvt.IA_readSource('Digit.png', True)
#imageMorph = fcvt.IP_imageMorph(imageDigit, 'erosion', (3,3), True)
#imageMorph = fcvt.IP_imageMorph(imageDigit, 'dilation', (5,5), True)
#imageMorph = fcvt.IP_imageMorph(imageDigit, 'opening', (3,3), True)
#imageMorph = fcvt.IP_imageMorph(imageDigit, 'closing', (5,5), True)
#
##Image filtering
#imageCameraman = fcvt.IA_readSource('cameraman_noise.jpg', True)
#imageFiltered = fcvt.IP_imageFilt(imageCameraman, 'average', (10,10), True)
#imageFiltered = fcvt.IP_imageFilt(imageCameraman, 'gaussian', (5,5 ), True)
#imageFiltered = fcvt.IP_imageFilt(imageCameraman, 'median', (3,3), True)
#
#
#
#
#"""""""""""""""""""""
#Feature Extraction
#
#"""""""""""""""""""""
##Color detction
#imageMap = fcvt.IA_readSource('map.png', True)
##lowerbound = [240,185,120]
##upperbound = [260,205,150]
#lowerbound = [120,185,240]
#upperbound = [150,205,260]
#imageColor = fcvt.FE_colorDetection(imageMap, lowerbound, upperbound, True)
#
##lowerbound = [100,205,245]
##upperbound = [170,230,265]
#lowerbound = [245,205,100]
#upperbound = [265,230,170]
#imageColor = fcvt.FE_colorDetection(imageMap, lowerbound, upperbound, True)
#
##lowerbound = [179,220,204]
##upperbound = [199,240,224]
#lowerbound = [204,220,179]
#upperbound = [224,240,199]
#imageColor = fcvt.FE_colorDetection(imageMap, lowerbound, upperbound, True)
#
##lowerbound = [228,214,235]
##upperbound = [248,254,255]
#lowerbound = [235,214,228]
#upperbound = [255,254,248]
#imageColor = fcvt.FE_colorDetection(imageMap, lowerbound, upperbound, True)
#
##Edge detection
#imageChessboard = fcvt.IA_readSource('imageChessboard.png', True)
#imageEdge = fcvt.FE_edgeDetection(imageChessboard, 'canny', True)
#
##Corner detection
#imageCorner = fcvt.FE_cornerDetection(imageChessboard, True)
#
##Local Binary Pattern (LBP)
#imageLBP = fcvt.FE_LBPDetection(image, True)
#
##Keypoint detection
#imageKeyPoint = fcvt.FE_keypointDetection(image, 'SIFT', True)
#imageKeyPoint = fcvt.FE_keypointDetection(image, 'SURF', True)
#
#
#
#"""""""""""""""""""""
#Feature Representation
#"""""""""""""""""""""
#cluster = fcvt.FE_Clustering(imageKeyPoint[1], 'Fuzzy', 5)
#
#quantisation = fcvt.FE_Quantisation(imageKeyPoint[1], 'Fuzzy', cluster[0])




"""""""""""""""""""""
Image classification
""""""""""""""""""""" 
#Output_crisp = fcvt.Image_Classification('Training_doc', 'Testing_doc', 'SIFT', 'Crisp')
#Output_fuzzy = fcvt.Image_Classification('Training_doc', 'Testing _doc', 'SIFT', 'Fuzzy')

#Output_fuzzy = fcvt.Image_Classification('Training_doc', 'Testing_doc', 'LBP', 'Crisp')
Output_fuzzy = fcvt.Image_Classification('Training_doc', 'Testing_doc', 'LBP', 'Fuzzy')












