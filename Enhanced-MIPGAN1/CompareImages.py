# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 05:25:18 2022

@author: ajava
"""

from deepface import DeepFace
import cv2
import pandas as pd
import numpy as np
import glob
from HistogramComparision import HistogramComparision
import piq
import torch
from torchvision import transforms
from PIL import Image
import os
from datetime import datetime

class CompareImages():
    def __init__(self):
        self.image1 = ""
        self.image2 = ""
        self.hist = 0
        self.mse = 0
        self.mssim = 0
        self.fsim = 0
        self.arcface_dist = 0
        self.arcface_acceptance = ""
        self.model = 0
        
    def calc_histogram(self):
        hr = HistogramComparision()
        self.hist = hr.comp2hist(self.image1, self.image2)
    
    def calc_mse(self):
        image1 = cv2.imread(self.image1)
        image2 = cv2.imread(self.image2)
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        self.mse = np.sum((gray1.astype("float") - gray2.astype("float")) ** 2)
        self.mse /= float(gray1.shape[0] * gray1.shape[1])
      
    def calc_mssim(self):
        loss = piq.MultiScaleSSIMLoss(data_range=1., reduction='none')
        convert_tensor = transforms.ToTensor()
        self.mssim = loss(torch.reshape(convert_tensor(Image.open(self.image1)), (1,3,1024,1024)), torch.reshape(convert_tensor(Image.open(self.image2)), (1,3,1024,1024))).item()
    
    def calc_fsim(self):
        loss = piq.FSIMLoss(data_range=1., reduction='none')
        convert_tensor = transforms.ToTensor()
        self.fsim = loss(torch.reshape(convert_tensor(Image.open(self.image1)), (1,3,1024,1024)), torch.reshape(convert_tensor(Image.open(self.image2)), (1,3,1024,1024))).item()
    
    def calc_arcface(self):
        arcs_dist = []
        arcs_verify = []
        for _ in range(5):
            arcface = DeepFace.verify(self.image1, self.image2, model_name="ArcFace", model=self.model)
            arcs_dist.append((arcface['distance']))
            arcs_verify.append(arcface['verified'])
        self.arcface_dist = min(set(arcs_dist))
        #self.arcface_dist = max(set(arcs_dist), key = arcs_dist.count)
        self.arcface_acceptance = arcs_verify[arcs_dist.index(self.arcface_dist)]

    def calc_all(self):
        self.calc_histogram()
        self.calc_mse()
        self.calc_mssim()
        self.calc_fsim()
        self.calc_arcface()
        return self.hist, self.mse, self.mssim, self.fsim, self.arcface_dist, self.arcface_acceptance
    
    def compare(self, datasets_path, references_path, model):       
        headers = ["refrence", "source", "hist diff", "mse", "mssim", "fsim", "arcface", "accepted"]
        now = datetime.now()
        print("started at: {}:{}:{}".format(now.hour, now.minute, now.second))
        self.model = model
        for dataset_path in datasets_path:
            for reference_path in references_path:
                folders = [name for name in os.listdir(dataset_path) 
                           if os.path.isdir(os.path.join(dataset_path, name))]
                for i in range(len(folders)):        
                    results = []     
                    for refPath in glob.glob(reference_path + "/*.png"):
                        for imagePath in glob.glob(dataset_path + folders[i] + '/save_result' + "/*.png"):
                            self.image1 = refPath
                            self.image2 = imagePath
                            #hist, mse, mssim, fsim, arcface_dist, arcface_accepance
                            images = [refPath.split('/')[-1], imagePath.split('/')[-1]]
                            values = self.calc_all()
                            row = [*images, *values]
                            results.append(row)

                    now = datetime.now()
                    print("config {} is done at: {}:{}:{}".format(i+1, now.hour, now.minute, now.second))
                    data = pd.DataFrame(results, columns=headers)
                    data.to_csv(dataset_path +'/ComparisionResults{}.csv'.format(i), index=False)
                    
            now = datetime.now()
            print("dataset done at: {}:{}:{}".format(now.hour, now.minute, now.second))
            
        now = datetime.now()
        print("ended at: {}:{}:{}".format(now.hour, now.minute, now.second))