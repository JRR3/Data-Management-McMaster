#JRR @ McMaster University
#Update: 18-Sep-2022
import numpy as np
import pandas as pd
import os
import re
import matplotlib as mpl
mpl.rcParams['figure.dpi']=300
import matplotlib.pyplot as plt
import datetime


# <b> Section: Sample Inventory </b>

# Columns B:E are the vaccination dates.
# We have this information from the Master Participant Data file.
# Column F is labeled Vaccine.
# This is also available in the Master Participant Data File.
# Columns G:V are Dried Blood Spot
# Columns W:END are Blood Draw

class SampleInventoryData:
    def __init__(self, path, df=None):

        self.merge_column = 'ID'

        if df is not None:
            self.initialize_class_with_df(df)
            print('SID class initialization from DF.')
        else:
            #Which row contains the first data entry in the Excel file
            self.excel_starts_at = 5
            self.dpath = path
            SID = 'Sample_Inventory_Data'
            self.fname = os.path.join(self.dpath, SID + '.xlsx')
            #Read the Excel file containing the data
            self.df = pd.read_excel(self.fname,
                                    sheet_name="All Sites - AutoFill",
                                    skiprows=[0,1,2])
            print('SID file has been loaded from Excel.')


    def initialize_class_with_df(self, df):
        self.df = df

    def rename_merging_column(self):
        dc = {'Sample ID': self.merge_column}
        self.df.rename(columns = dc, inplace=True)

    def remove_zeros(self):
        #Some dates are the product of having zeros in Excel.
        #We convert this dates to NaN.
        self.df.replace(datetime.time(0,0), np.nan, inplace=True)


    def check_for_repeats_in_sample_inventory(self):
        #We drop rows containing NA in the "ID" from the
        #file because sometimes there are additional comments
        #below the table that get interpreted as additional rows.
        self.df.dropna(subset=[self.merge_column], inplace=True)
        value_count_gt_1 = self.df[self.merge_column].value_counts().gt(1)
        if value_count_gt_1.any():
            print('We have repetitions in Sample Inventory')
            raise ValueError('No repetitions were expected.')
        else:
            print('No repeats were found in Sample Inventory')


    def relabel_and_delete_columns(self):
        #Relabel the columns to indicate if the sample is DBS or Blood Draw.
        #Delete columns related to vaccine and the N-th dose
        dc = {}
        block_to_delete_starts_at = 1
        block_to_delete_ends_at  = -1
        dbs_start_partial_label = 'v-e'
        bd_start_partial_label = 'baseline - b'
        dbs_start_index = -1
        bd_start_index  = -1
        flag_1 = True
        flag_2 = False
        modifier = ''
        L = []
        #First we relabel
        for k,column in enumerate(self.df.columns):
            if flag_1 and dbs_start_partial_label in column.lower():
                dbs_start_index = k
                block_to_delete_ends_at = k
                flag_1 = False
                flag_2 = True
                modifier = 'DBS:'
            elif flag_2 and bd_start_partial_label in column.lower():
                bd_start_index = k
                flag_2 = False
                modifier = 'Blood Draw:'
            dc[column] = modifier + column
        #Rename columns
        self.df.rename(columns = dc, inplace=True)
        #Delete columns
        R = np.arange(block_to_delete_starts_at,
                      block_to_delete_ends_at,
                      dtype=int)
        self.df.drop(self.df.columns[R], axis = 1, inplace=True)



    def print_col_names_and_types(self):
        for name, dtype in zip(self.df.columns, self.df.dtypes):
            print(name, ':', dtype)


    def generate_excel_file(self):
        fname = 'Sample_Inventory_Data_X.xlsx'
        fname = os.path.join(self.dpath, fname)
        self.df.to_excel(fname, index=False)
        print('Excel file was produced.')

    def compare_data_frames(self):
        df1 = pd.read_excel('./Sample_Inventory_Data_X.xlsx')
        df2 = pd.read_excel('./Sample_Inventory_Data_Y.xlsx')
        if df1.equals(df2):
            print('They are equal')

    def load_main_frame(self):
        fname = 'Sample_Inventory_Data_X.xlsx'
        fname = os.path.join(self.dpath, fname)
        self.df = pd.read_excel(fname)
        print('Excel file was loaded.')

    def update_master(self, df_m, kind='full'):
        #Note that this function can be generalized.
        #This function updates the master data frame
        #df_m, which is the result of merging
        #(1) The Master Participant Data file
        #(2) The LTC Infection Summary file
        #(3) The Sample Inventory File
        #First, we read the update and 
        #give it the right format.
        self.full_run()
        #Check compatibility of columns
        original_list_of_columns = df_m.columns
        for column in self.df:
            if column not in df_m.columns:
                raise ValueError('Incompatible columns in SID')
        print('All columns are compatible')
        #Left = Master DF
        #Right= Update
        M = pd.merge(df_m, self.df, on=self.merge_column, how='outer')
        for column in self.df:
            if column == self.merge_column:
                #Ignore the ID column
                continue
            left = column + '_x'
            right = column + '_y'
            if kind == 'full':
                #Only trust the update
                M[column] = M[right]
            elif kind == 'complement':
                #Keep the original if not empty.
                #Otherwise, use the update.
                M[column] = M[left].where(M[left].notnull(), M[right])
            else:
                raise ValueError('Unexpected kind for the SID update')
            #Erase the left and right columns
            M.drop(columns = [left, right], inplace=True)

        #Use the original order of the columns
        return M[original_list_of_columns]


    def get_last_blood_draw(self, ID):
        selector = self.df['ID'] == ID
        row = self.df.loc[selector]
        if 1 < len(row):
            print(f'{ID=}')
            raise ValueError('Unexpected repetitions. ID is not unique.')
        L = []
        #We assume the rows have a chronological order.
        for col in self.df.columns:
            if col.lower().startswith('blood draw'):
                entry = row[col]
                #print(f'{entry}')
                if entry.notnull().all():
                    value = entry.values[0]
                    if isinstance(value, datetime.datetime):
                        L.append(value)
        if 0 == len(L):
            return None
        else:
            return L[-1]







    def full_run(self):
        self.rename_merging_column()
        self.remove_zeros()
        self.check_for_repeats_in_sample_inventory()
        self.relabel_and_delete_columns()
        self.generate_excel_file()
        #self.print_col_names_and_types()
        #self.compare_data_frames()
        print('Module: module_Sample_Inventory.py FINISHED')
