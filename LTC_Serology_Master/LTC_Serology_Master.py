#JRR @ McMaster University
#Update: 18-Sep-2022
import numpy as np
import pandas as pd
import os
import re
import datetime
import shutil
import matplotlib as mpl
mpl.rcParams['figure.dpi']=300
import matplotlib.pyplot as plt


# <b> Section: Master_Participant_Data </b>

class LTCSerologyMaster:
    def __init__(self, dpath, parent=None):
        self.parent = None
        self.df     = None
        self.dpath  = dpath
        self.backups_path = os.path.join('..', 'backups')
        self.merge_source = 'Full ID'
        self.merge_column = 'ID'
        self.DOC          = 'Date Collected'

        self.load_LSM_file()

        if parent:
            self.parent = parent
            print('LSM class initialization from Manager.')
        else:
            raise ValueError('Parent object is unavailable.')

    def load_LSM_file(self):
        #Note that the LSM file already has compact format, i.e.,
        #the headers are simple.
        fname = 'LSM.xlsx'
        fname = os.path.join(self.dpath, fname)
        #Read the Excel file containing the data
        self.df = pd.read_excel(fname)

        print('LSM class has been initialized with LSM file.')


    def update_id_column(self):
        #This function was updated on Oct 31, 2022
        self.df = self.parent.create_df_with_ID_from_full_ID(self.df)

    ######September 28 20222

    def backup_the_LSM_file(self):
        fname = 'LSM.xlsx'
        original = os.path.join(self.dpath, fname)
        today = datetime.datetime.now()
        date  = today.strftime('%d_%m_%Y_time_%H_%M_%S')
        bname = 'LSM' + '_backup_' + date
        bname += '.xlsx'
        backup   = os.path.join(self.backups_path, bname)
        shutil.copyfile(original, backup)
        print('A backup for the LSM file has been generated.')

    def write_LSM_to_excel(self):
        self.backup_the_LSM_file()
        fname = 'LSM.xlsx'
        fname = os.path.join(self.dpath, fname)
        print('Writing the LSM file to Excel.')
        self.df.to_excel(fname, index = False)
        print('The LSM file has been written to Excel.')

    def merge_serology_update(self, df_up):
        #Updated on Oct 31, 2022
        print('===================Work======')
        df_up.replace('NT', np.nan, inplace=True)
        relevant_proteins = ['Spike', 'RBD', 'Nuc']
        relevant_Igs      = ['IgG', 'IgA', 'IgM' ]
        rexp_n = re.compile('/[ ]*(?P<dilution>[0-9]+)')
        rexp_c = re.compile('[0-9]+[-][0-9]+[-][a-zA-Z]+')
        col_indices = {}
        exit_iterrows_flag = False
        row_start = -1
        id_col_index = -1
        list_of_proteins = []
        list_of_Igs = []
        list_of_dilutions = []
        for index, row in df_up.iterrows():
            if exit_iterrows_flag:
                break
            for col, item in row.items():
                if isinstance(item, str):
                    #Check if the item is an ID
                    obj = rexp_c.search(item)
                    if obj is not None:
                        #print('Data start at row:', index)
                        id_col_index = col
                        row_start = index
                        exit_iterrows_flag = True
                        break
                    for protein in relevant_proteins:
                        if protein.lower() in item.lower():
                            col_indices[col] = None
                            list_of_proteins.append(protein)
                            break
                    for Ig in relevant_Igs:
                        if Ig.lower() in item.lower():
                            col_indices[col] = None
                            list_of_Igs.append(Ig)
                            break
                    #Check if the item is a dilution
                    obj = rexp_n.search(item)
                    if obj is not None:
                        dilution = obj.group('dilution')
                        col_indices[col] = None
                        list_of_dilutions.append(dilution)
        #Form headers
        for k, p, Ig, dil in zip(col_indices.keys(),
                                 list_of_proteins,
                                 list_of_Igs,
                                 list_of_dilutions):
            s = '-'.join([p,Ig,dil])
            col_indices[k] = s

        merge_at_column = 'Full ID'
        #Set in the dictionary the mapping:
        #id_col_index -> merge_at_column
        col_indices[id_col_index] = merge_at_column
        print(col_indices)
        df_up.rename(columns = col_indices, inplace = True)
        def is_id(txt):
            if txt is np.nan:
                return txt
            obj = rexp_c.search(txt)
            if obj:
                return obj.group(0)
            else:
                return np.nan
        df_up[merge_at_column] = df_up[merge_at_column].apply(is_id)
        df_up.dropna(subset=merge_at_column, axis=0, inplace=True)
        print('Ready to merge')
        #Check that the use of the dictionary can be replaced
        #with the modified data frame df_up
        #for key, value in col_indices.items()
        #Merge process >>>
        #The update has a higher priority than the original data.
        kind = 'update+'
        self.df = self.parent.merge_X_with_Y_and_return_Z(self.df,
                                                          df_up,
                                                          merge_at_column,
                                                          kind=kind)
        self.update_id_column()
        print('End of merging process.')

    def update_LND_data(self):
        #Updated on Nov 10, 2022
        fname  = 'LND_update.xlsx'
        folder = 'Jessica_2_nov_10_2022'
        fname = os.path.join('..','requests',folder, fname)
        df_up = pd.read_excel(fname)
        df_up.dropna(axis=0, how='all', inplace=True)
        print(df_up)
        kind = 'original+'
        merge_at_column = 'Full ID'
        self.df = self.parent.merge_X_with_Y_and_return_Z(self.df,
                                                          df_up,
                                                          merge_at_column,
                                                          kind=kind)
