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
    def __init__(self, dpath):
        self.dpath = dpath
        self.backups_path = os.path.join('..', 'backups')
        self.merge_source = 'Full ID'
        self.merge_column = 'ID'

        self.load_LSM_file()

    def load_LSM_file(self):
        #Note that the LSM file already has compact format, i.e.,
        #the headers are simple.
        fname = 'LSM.xlsx'
        fname = os.path.join(self.dpath, fname)
        #Read the Excel file containing the data
        self.df = pd.read_excel(fname)

        print('LSM class has been initialized with LSM file.')

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
        column_order = [self.merge_column] + self.df.columns.to_list()
        self.df[self.merge_column] = L
        self.df = self.df[column_order]
        print('ID column has been created inside the LSM file.')
        print('Columns should be in the original order.')
        #print(self.df)

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

    def write_to_excel(self):
        self.backup_the_LSM_file()
        fname = 'LSM.xlsx'
        fname = os.path.join(self.dpath, fname)
        self.df.to_excel(fname, index = False)
        print('The LSM file has been written to Excel.')

    def merge_serology_update(self, df_up):
        original_columns = self.df.columns
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
            for col, item in row.iteritems():
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
        prioritize_update = True
        M = pd.merge(self.df, df_up, on=merge_at_column, how='outer')
        for key, value in col_indices.items():
            if value == merge_at_column:
                continue
            if value in self.df.columns:
                left = value + '_x'
                right = value + '_y'
                z = M[left].isnull() &  M[right].notnull()
                if z.any():
                    print('New indeed for', value)
                if value in M.columns:
                    raise ValueError('This column should not exist.')
                if prioritize_update:
                    print('The update has priority over the current data.')
                    M[value] = M[right].where(M[right].notnull(), M[left])
                else:
                    print('The current data has priority over the update.')
                    M[value] = M[left].where(M[left].notnull(), M[right])
                #Drop repeated columns
                M.drop([left, right], axis=1, inplace=True)
            else:
                raise ValueError('This column does not exist.')
        self.df = M[original_columns]
        print('End of merging process.')


