# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 18:03:49 2021

@author: Ricky
"""

import six
import os  # needed navigate the system to get the input data
import radiomics
from radiomics import featureextractor
import SimpleITK as sitk
import numpy as np
import nrrd
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from glob import glob
import pandas as pd


#returns list of radiomic features for each mask
def computeRadiomicFeatures(imPath, maskThresh):
    
    #getting paths of directories and images
    imName = os.path.splitext(os.path.basename(imPath))[0]
    imDir = os.path.dirname(imPath)

    imGray = cv2.imread(imPath, cv2.IMREAD_GRAYSCALE)
    imMask = np.where(imGray>maskThresh, 1, 0)
    
    #writing to .nrrd because pyradiomics only takes in nrrd or SITK (??? WHY ???)
    #HAVE TO SAVE AS NRRD FIRST, because SITK is terrible at 
    #handling 3-channel RGB images (can try with SITK after but NRRD is the
    #most "proper" physics since it's from a matrix of grayscale intensities)
    imNrrdDir = (imDir + r'\\imNrrdFiles\\')
    maskNrrdDir = (imDir + r'\\maskNrrdFiles\\')
    imNrrdPath = (imNrrdDir + imName + '.nrrd')
    maskNrrdPath = (maskNrrdDir + imName + '.nrrd')
    

    if not(os.path.exists(maskNrrdDir)):
        os.mkdir(maskNrrdDir)
    if not(os.path.exists(imNrrdDir)):
        os.mkdir(imNrrdDir)
        
    nrrd.write(maskNrrdPath, imMask)
    nrrd.write(imNrrdPath, imGray)
    
    
    #defining radiomics settings
    settings = {}
    settings['force2D'] = True
    #instantiating extractor, 2D shape disabled by default so we have to enable
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableFeatureClassByName('shape2D')
    #print('Extraction parameters:\n\t', extractor.settings)
    #print('Enabled filters:\n\t', extractor.enabledImagetypes)
    #print('Enabled features:\n\t', extractor.enabledFeatures)
    
    featureListAllMasks = []
    featureNames = []
    featureNames.extend(['imName'])
    

 
    if(sum(map(sum, imMask))==0):
        #blank 103-item array
        featureList = []
        featureList.extend([imName])
        featureList.extend(102*[0])
    else:
        result = extractor.execute(imNrrdPath, maskNrrdPath)
    
        featureList = []
        featureList.extend([imName])
        for key, value in six.iteritems(result):
            #ignore metadata
            if('diagnostics') not in key:
                if (type(value) == np.ndarray ):
                    featureList.append(value.item())
                else:
                    featureList.append(value)
     
    featureListAllMasks.append(featureList)
        
    #after last run, create the list of feature names if there was a segmentation
    
    if(sum(map(sum, imMask))==0):
        return featureListAllMasks, 'empty'
    else:
        allKeys = result.keys()
        featureNamesTemp = [item for item in allKeys if ('diagnostics' not in item)]
        featureNames.extend(featureNamesTemp)
                
        return featureListAllMasks, featureNames
   
#---------------------------------------------------------------------    
#---------------------------------------------------------------------   
# example of how to run script here


maskThresh = 1;
dataDir =  r"C:\Users\rckyh\Desktop\repo\carotid-ultrasound\\"
imDir =  dataDir + "masks\\"
output = dataDir + 'radiomics.csv'


featureListAllImages = []
imList = []
imList.extend(glob(os.path.join(imDir, "*.jpg")))
imList.extend(glob(os.path.join(imDir, "*.tif")))
for imPath in imList:
    print(imPath)
    featureList, tempFeatureNames = computeRadiomicFeatures(imPath, maskThresh)

    if(tempFeatureNames != 'empty'):
        featureNames = tempFeatureNames
        
    featureListAllImages.extend(featureList)


print('Saving: ', output)
statsDF = pd.DataFrame(featureListAllImages, columns = featureNames)
statsDF.to_csv(output, index=False)
