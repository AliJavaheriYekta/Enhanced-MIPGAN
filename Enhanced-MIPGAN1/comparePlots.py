# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 01:58:07 2022

@author: ajava
"""

import pandas as pd
import math
import matplotlib.pyplot as plt
import glob
import os

class PlotResults():
    def __init__(self):
        self.df1 = []
        self.df2 = []
    
    def plotResults(self,path):
        cnt = 1
        for compFile in glob.glob(path + "/*.csv"):
            folder = os.path.join(path, 'Results{}plots'.format(cnt))
            os.mkdir(folder)
            df = pd.read_csv(compFile)
            keys = df.keys()[2:-1]
            df1 = df[: math.ceil(len(df)/2)]
            df2 = df[math.ceil(len(df)/2) :]
            
            for i in range(len(df)):
                if i in range(len(df1)):
                    df1.loc[i,'source'] = int(df1.loc[i,'source'].split('.png')[0])
                else:
                    df2.loc[i,'source'] = int(df2.loc[i,'source'].split('.png')[0])
            
            self.df1 = df1.sort_values('source')
            self.df2 = df2.sort_values('source')
            cnt = cnt + 1
            for key in keys:
                self.plot(key, folder)
    
    def plot(self, key, folder):
        x = self.df1['source'].tolist() 
        y = self.df1[key].tolist()
    
        xx = self.df2['source'].tolist()
        yy = self.df2[key].tolist()
    
        plt.plot(x, y)
        plt.plot(xx, yy, 'r-')
    
        plt.xlabel('morph iters')
        plt.ylabel(key)
        plt.legend( [ 'first reference'   # This is f(x)
                    , 'second reference'       # This is g(x)
                    ] )
        plt.title('{} reaction to morphing'.format(key));
        plt.savefig(folder + '/{}.png'.format(key))
        plt.clf()