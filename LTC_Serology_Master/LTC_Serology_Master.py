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
        self.ID_columns   = [self.merge_source,
                self.merge_column]
        self.DOC          = 'Date Collected'
        self.non_numeric_columns = []
        self.numeric_columns     = []

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
        self.non_numeric_columns = self.ID_columns + [self.DOC]
        for column in self.df.columns:
            if column not in self.non_numeric_columns:
                self.numeric_columns.append(column)

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

        #Full ID
        merge_at_column = self.merge_source
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
        #Remove individuals with "E" type label.
        self.remap_E_type_individuals(df_up)
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

    def remap_E_type_individuals(self, df_up):
        #Careful with the "E" type individuals.
        fname  = 'remapping_list.xlsx'
        fname = os.path.join(self.dpath, fname)
        df_re = pd.read_excel(fname)
        for index_re, row_re in df_re.iterrows():
            old_id = row_re['Original']
            new_id = row_re['New']
            df_up[self.merge_source].replace(old_id, new_id, inplace=True)

    def update_LND_data(self):
        #Updated on Nov 10, 2022
        fname  = 'LND_update.xlsx'
        folder = 'Jessica_2_nov_10_2022'
        fname = os.path.join('..','requests',folder, fname)
        df_up = pd.read_excel(fname)
        #If no Full ID is provided, the row is removed.
        df_up.dropna(axis=0, subset=self.merge_source, inplace=True)
        #Remap faulty "Full IDs"
        self.remap_E_type_individuals(df_up)
        kind = 'original+'
        merge_at_column = 'Full ID'
        self.df = self.parent.merge_X_with_Y_and_return_Z(self.df,
                                                          df_up,
                                                          merge_at_column,
                                                          kind=kind)
    def what_is_missing(self):
        col_to_ids = {}
        max_count = -1
        for column in self.df.columns:
            if column in self.non_numeric_columns:
                continue
            selection = self.df[column].isnull()
            n_of_empty_cells = selection.value_counts()[True]
            if max_count < n_of_empty_cells:
                max_count = n_of_empty_cells
            df_s = self.df[selection]
            IDs  = df_s[self.merge_source].values
            col_to_ids[column] = IDs
        print(f'{max_count=}')
        for key, L in col_to_ids.items():
            n_of_empty_cells = len(L)
            n_nans = max_count - n_of_empty_cells
            col_to_ids[key] = np.pad(L, (0,n_nans), 'empty')
        df_m = pd.DataFrame(col_to_ids)
        label = 'missing_data_for_serology_samples'
        self.parent.write_df_to_excel(df_m, label)


