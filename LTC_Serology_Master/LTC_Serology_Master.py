#JRR @ McMaster University
#Update: 18-Sep-2022
import numpy as np
import pandas as pd
import os
import re
import matplotlib as mpl
mpl.rcParams['figure.dpi']=300
import matplotlib.pyplot as plt


# <b> Section: Master_Participant_Data </b>

class LTCSerologyMaster:
    def __init__(self, path):
        self.dpath = path
        #Note that the LSM file already has compact format
        LSM = 'LSM'
        self.fname = os.path.join(self.dpath, LSM + '.xlsx')
        #Read the Excel file containing the data
        self.df = pd.read_excel(self.fname)
        self.merge_source = 'Full ID'
        self.merge_column = 'ID'
        print('LTCSerologyMaster class has been loaded.')

        self.create_id_column()

    def create_id_column(self):
        rx = re.compile('[0-9]+[-][0-9]+')
        L = []
        for index, row in self.df.iterrows():
            txt = row[self.merge_source]
            obj = rx.match(txt)
            if obj is None:
                raise ValueError('ID is not compliant.')
            L.append(obj.group(0))
        if self.df[self.merge_source].value_counts().gt(1).any():
            raise ValueError('No repetitions should be present.')
        self.df[self.merge_column] = L
        print('ID column has been created inside the LSM file.')

