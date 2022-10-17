#JRR @ McMaster University
#Update: 03-Oct-2022
import sys
import os
import re
import pandas as pd
import numpy as np
import shutil
import datetime



class LTCNeutralizationData:
    def __init__(self, dpath, df=None):

        self.df = None
        self.dpath = dpath
        self.backups_path = os.path.join('..', 'backups')
        self.merge_source = 'Full ID'
        self.merge_column = 'ID'

        self.load_LND_file()

    def load_LND_file(self):
        #Note that the LND file already has compact format, i.e.,
        #the headers are simple.
        fname = 'LND.xlsx'
        fname = os.path.join(self.dpath, fname)
        #Read the Excel file containing the data
        self.df = pd.read_excel(fname)
        print('LND class has been initialized with LND file.')

    def clean_LND_file(self):
        self.df.replace('.', np.nan, inplace=True)
        #self.df.replace('#REF!', np.nan, inplace=True)
        #self.df['Date Collected'] = pd.to_datetime(self.df['Date Collected'])
        #print(self.df['Date Collected'])
        print(self.df)


