#JRR @ McMaster University
#Update: 31-Oct-2022
import sys
import os
import re
import pandas as pd
import numpy as np
import shutil
import datetime



class LTCNeutralizationData:
    def __init__(self, dpath, parent=None):

        self.parent = None
        self.df     = None
        self.dpath  = dpath
        self.backups_path = os.path.join('..', 'backups')
        self.merge_source = 'Full ID'
        self.merge_column = 'ID'

        self.load_LND_file()

        if parent:
            self.parent = parent
            print('LND class initialization from Manager.')
        else:
            raise ValueError('Parent object is unavailable.')

    def load_LND_file(self):
        #Note that the LND file already has compact format, i.e.,
        #the headers are simple.
        fname = 'LND.xlsx'
        fname = os.path.join(self.dpath, fname)
        #Read the Excel file containing the data
        self.df = pd.read_excel(fname)
        print('LND class has been initialized with LND file.')
        print(f'LND file has {len(self.df)} rows.')

    def clean_LND_file(self):
        self.df.replace('.', np.nan, inplace=True)
        print(self.df)

    def backup_the_LND_file(self):
        fname = 'LND.xlsx'
        original = os.path.join(self.dpath, fname)
        today = datetime.datetime.now()
        date  = today.strftime('%d_%m_%Y_time_%H_%M_%S')
        bname = 'LND' + '_backup_' + date
        bname += '.xlsx'
        backup   = os.path.join(self.backups_path, bname)
        shutil.copyfile(original, backup)
        print('A backup for the LND file has been generated.')

    def write_to_excel(self):
        self.backup_the_LND_file()
        fname = 'LND.xlsx'
        fname = os.path.join(self.dpath, fname)
        print('Writing the LND file to Excel.')
        self.df.to_excel(fname, index = False)
        print('The LND file has been written to Excel.')

    def update_id_column(self):
        #This function was updated on Oct 31, 2022
        self.df = self.parent.create_df_with_ID_from_full_ID(self.df)


