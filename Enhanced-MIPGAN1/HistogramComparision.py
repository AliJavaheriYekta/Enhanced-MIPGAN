# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 21:43:28 2022

@author: ajava
"""

# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import math
import pandas as pd

#%%
# construct the argument parser and parse the arguments

class HistogramComparision():
    def __init__(self):
        self.OPENCV_METHODS = {
        	"Correlation": cv2.HISTCMP_CORREL,
        	"Chi-Squared": cv2.HISTCMP_CHISQR,
        	"Intersection": cv2.HISTCMP_INTERSECT,
        	"Hellinger": cv2.HISTCMP_BHATTACHARYYA}
        self.methodName = "Correlation"

    
    def comp2hist(self, image1, image2):
        image1 = cv2.cvtColor(cv2.imread(image1), cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(cv2.imread(image2), cv2.COLOR_BGR2RGB)
        hist1 = cv2.calcHist([image1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.calcHist([image2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.normalize(hist2, hist2).flatten()
        method = self.OPENCV_METHODS[self.methodName]
        difference = cv2.compareHist(hist1, hist2, method)
        return difference


