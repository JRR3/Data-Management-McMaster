#JRR @ McMaster University
#Update: 18-Sep-2022
import sys
import os
import re
import pandas as pd
import shutil
import datetime

    def load_the_M_file(self):
        #Load the merged file M
        print('Make sure that this is a clone of the encrypted version!')
        fname = self.M_fname
        fname = os.path.join(self.outputs_path, fname)
        self.df = pd.read_excel(fname)
        print('MPD_LIS_SID, aka the M file, has been loaded from Excel')
