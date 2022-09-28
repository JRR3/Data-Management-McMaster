#JRR @ McMaster University
#Update: 26-Sep-2022
import sys
import os
import re
import pandas as pd
import shutil
import datetime

class Comparator:
    def __init__(self):
        self.outputs_path = os.path.join('..', 'outputs')
        self.compare_path = os.path.join('..', 'compare')
        self.df = None

    def load_the_M_file(self):
        #Load the merged file M
        print('Make sure that this is a clone of the encrypted version!')
        fname = 'M.xlsx'
        fname = os.path.join(self.outputs_path, fname)
        self.df = pd.read_excel(fname)
        print('MPD_LIS_SID, aka the M file, has been loaded from Excel')

    def load_the_rainbow(self):
        fname = 'rainbow.xlsx'
        fname = os.path.join(self.compare_path, fname)
        df_rain = pd.read_excel(fname, skiprows = [0])
        print(df_rain)

    def satisfy_request(self):
        #This function extracts a slice of the main data frame.
        #It uses the IDs in the request file to produce the slice.
        #Requested on: 16_09_2022
        fname = 'request_16_Sep_2022.xlsx'
        fname = os.path.join(self.requests_path, fname)
        df_request = pd.read_excel(fname)
        print(f'{df_request.shape=}')
        #Do we have repetitions?
        value_counts = df_request['ID'].value_counts().gt(1)
        vc = value_counts.loc[value_counts]
        for individual, _ in vc.iteritems():
            print(f'Repeated {individual=}')
        #First, we make sure that all the requested individuals
        #are present in our database.
        in_database = df_request['ID'].isin(self.df['ID'])
        in_database_count = in_database.sum()
        print(f'{in_database_count=}')
        if not in_database.all():
            missing_individuals = df_request['ID'].loc[~in_database]
            for index, individual in missing_individuals.iteritems():
                print(f'{individual=} is missing')

        #Satisfy request with available individuals
        selection = self.df['ID'].isin(df_request['ID'])
        selection_sum = selection.sum()
        print(f'{selection_sum=}')
        S = self.df.loc[selection,:]

        fname = 'data_requested_on_16_Sep_2022.xlsx'
        fname = os.path.join(self.outputs_path, fname)
        S.to_excel(fname, index = False)

obj = Comparator()
obj.load_the_rainbow()
