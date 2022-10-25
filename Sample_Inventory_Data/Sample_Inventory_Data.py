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
    def __init__(self, path, parent=None):

        self.dpath = path
        self.merge_column = 'ID'
        self.blood_draw   = 'blood draw'
        self.blood_draw_code_to_col_name = {}
        self.original_to_current = {}
        self.code_regexp = re.compile('[-][ ]*(?P<code>[A-Z]+)')

        if parent:
            self.parent = parent
            self.relate_blood_draw_code_to_col_name()
            self.relate_original_column_names_to_current()
            print('SID class has been initialized with the Manager class.')
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



    def rename_merging_column(self):
        dc = {'Sample ID': self.merge_column}
        self.df.rename(columns = dc, inplace=True)

    def remove_zeros(self):
        #Oct 12, 2022
        #Some dates are the product of having zeros in Excel.
        #We convert these dates to NaN.
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

    #Oct 5 2022
    def extract_letter_code(self, txt):
        obj = self.code_regexp.search(txt)
        if obj:
            return obj.group('code')
        else:
            raise ValueError('Unable to extract letter code.')

    def relate_original_column_names_to_current(self):
        #Oct 12, 2022
        fname = 'label_dictionary.xlsx'
        fname = os.path.join(self.dpath, fname)
        df_up = pd.read_excel(fname)
        for index, row in df_up.iterrows():
            original = row['Original']
            current = row['Current']
            self.original_to_current[original] = current
        #print(df_up)

    def format_megans_update(self, df_up):
        #Rename columns
        df_up.rename(columns=self.original_to_current, inplace=True)
        #Replace 0-dates
        df_up.replace(datetime.time(0,0), np.nan, inplace=True)
        #Check for repeats
        df_up.dropna(subset=[self.merge_column], inplace=True)
        value_count_gt_1 = df_up[self.merge_column].value_counts().gt(1)
        if value_count_gt_1.any():
            print('We have repetitions in Sample Inventory')
            raise ValueError('No repetitions were expected.')
        else:
            print('No repeats were found in Sample Inventory')

    def relate_blood_draw_code_to_col_name(self):
        #This function uses the PARENT object.
        #Generate a dictionary to relate the
        #blood draw code to the column name in the M file.
        for col_name in self.parent.df.columns:
            if col_name.lower().startswith(self.blood_draw):
                code = self.extract_letter_code(col_name)
                self.blood_draw_code_to_col_name[code] = col_name
        #print(self.blood_draw_code_to_col_name)



    def check_LSM_dates_using_SID(self):
        #This function was modified on Oct 12, 2022
        #This function uses the PARENT object.
        #Check the dates in the LSM '
        #file using the SID file.
        #Notation
        #doc: date of collection
        print('Checking the dates of the LSM file using the SID.')
        doc = 'Date Collected'
        missing_dates = []
        for index, row in self.parent.LSM_obj.df.iterrows():
            ID = row['ID']
            full_ID = row['Full ID']
            letter_code = self.extract_letter_code(full_ID)
            col_name_for_code = self.blood_draw_code_to_col_name[letter_code]
            #Note that the doc cannot be empty because
            #it is connected to the letter code portion
            #of the full id.
            doc_LSM = row[doc]
            selector = self.parent.df['ID'] == ID
            doc_M = self.parent.df.loc[selector, col_name_for_code]
            if doc_M.isnull().any():
                print(f'{full_ID=}')
                missing_dates.append(full_ID)
                continue
                #raise ValueError('The date for this ID DNE in the M file.')
            doc_M = doc_M.values[0]
            if pd.notnull(doc_LSM):
                delta = (doc_M - doc_LSM) / np.timedelta64(1, 'D')
                delta = np.abs(delta)
                if 1 < delta:
                    raise ValueError('The date for this ID does not match the M file.')
            else:
                #If the date is empty in the LSM file, we 
                #simply complete it with the M file.
                print(f'{full_ID=} DOC was assigned from SID.')
                self.parent.LSM_obj.df.loc[index, doc] = doc_M
        if  0 < len(missing_dates):
            print('The following Full IDs have missing dates:')
            S = pd.Series(missing_dates)
            print(S.to_string(index=False))
        else:
            print('All dates in the LSM file are consistent.')
        #self.parent.LSM_obj.write_to_excel()



