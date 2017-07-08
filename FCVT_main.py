# -*- coding: utf-8 -*-
"""
@author: ChernHong Lim
"""

"""
Main testing
"""

import FCVT as fcvt
  
"""""""""""""""""""""
Image Acquisition
"""""""""""""""""""""
#Provide setting
sourceDir = 'im_lenna.png' 

#Read image
image = fcvt.IA_readSource(sourceDir, True)


"""""""""""""""""""""
Image Preprocessing
"""""""""""""""""""""

#Image resize
imageResize = fcvt.IP_resize(image, 0.5, 0.5, True)

#Convert image to gray scale
imageGray = fcvt.IP_convertGray(image, True)
imageBinary = fcvt.IP_convertBinary(image, True)

#Image Morphological Operation
imageDigit = fcvt.IA_readSource('im_digit.png', True)
imageMorph = fcvt.IP_imageMorph(imageDigit, 'erosion', (3,3), True)
imageMorph = fcvt.IP_imageMorph(imageDigit, 'dilation', (5,5), True)
imageMorph = fcvt.IP_imageMorph(imageDigit, 'opening', (3,3), True)
imageMorph = fcvt.IP_imageMorph(imageDigit, 'closing', (5,5), True)

#Image filtering
imageCameraman = fcvt.IA_readSource('im_cameraman_noise.jpg', True)
imageFiltered = fcvt.IP_imageFilt(imageCameraman, 'average', (10,10), True)
imageFiltered = fcvt.IP_imageFilt(imageCameraman, 'gaussian', (5,5 ), True)
imageFiltered = fcvt.IP_imageFilt(imageCameraman, 'median', (3,3), True)




"""""""""""""""""""""
Feature Extraction

"""""""""""""""""""""
#Color detction
imageMap = fcvt.IA_readSource('im_map.png', True)
#lowerbound = [240,185,120]
#upperbound = [260,205,150]
lowerbound = [120,185,240]
upperbound = [150,205,260]
imageColor = fcvt.FE_colorDetection(imageMap, lowerbound, upperbound, True)

#lowerbound = [100,205,245]
#upperbound = [170,230,265]
lowerbound = [245,205,100]
upperbound = [265,230,170]
imageColor = fcvt.FE_colorDetection(imageMap, lowerbound, upperbound, True)

#lowerbound = [179,220,204]
#upperbound = [199,240,224]
lowerbound = [204,220,179]
upperbound = [224,240,199]
imageColor = fcvt.FE_colorDetection(imageMap, lowerbound, upperbound, True)

#lowerbound = [228,214,235]
#upperbound = [248,254,255]
lowerbound = [235,214,228]
upperbound = [255,254,248]
imageColor = fcvt.FE_colorDetection(imageMap, lowerbound, upperbound, True)

#Edge detection
imageChessboard = fcvt.IA_readSource('im_imageChessboard.png', True)
imageEdge = fcvt.FE_edgeDetection(imageChessboard, True)

#Corner detection
imageCorner = fcvt.FE_cornerDetection(imageChessboard, True)

#Local Binary Pattern (LBP)
imageLBP = fcvt.FE_LBPDetection(image, True)

#Keypoint detection
imageKeyPoint = fcvt.FE_keypointDetection(image, 'SIFT', True)
imageKeyPoint = fcvt.FE_keypointDetection(image, 'SURF', True)



"""""""""""""""""""""
Feature Representation
"""""""""""""""""""""
cluster = fcvt.FE_Clustering(imageKeyPoint[1], 5)
#
descriptor = fcvt.FE_Quantisation(imageKeyPoint[1], cluster[0])




"""""""""""""""""""""
Image classification
""""""""""""""""""""" 
Output_crisp = fcvt.Image_Classification('Training', 'Testing', 'SIFT', 'Crisp')
Output_fuzzy = fcvt.Image_Classification('Training', 'Testing2', 'SIFT', 'Fuzzy')

Output_fuzzy = fcvt.Image_Classification('Training', 'Testing', 'LBP', 'Crisp')
Output_fuzzy = fcvt.Image_Classification('Training', 'Testing', 'LBP', 'Fuzzy')












