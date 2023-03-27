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
        self.i_blood_draw_code_to_col_name = {}
        self.original_to_current = {}
        self.code_regexp = re.compile('[-][ ]*(?P<code>[A-Z]+[0-9]*)')

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
        #This function related the headers in Megan's
        #file with the labels in the master file.
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
        #Drop empty rows
        df_up.dropna(subset=[self.merge_column], axis=0, inplace=True)
        #Check ID format
        self.parent.check_id_format(df_up, self.merge_column)
        #Replace old IDs with new
        self.parent.MPD_obj.map_old_ids_to_new(df_up)
        #Check for repeats
        value_count_gt_1 = df_up[self.merge_column].value_counts().gt(1)
        if value_count_gt_1.any():
            print('We have repetitions in Sample Inventory')
            print(df_up[self.merge_column].value_counts())
            raise ValueError('No repetitions were expected.')
        else:
            print('No repeats were found in Sample Inventory')

        #Keep only dates and ID
        for index_r, row in df_up.iterrows():
            #print('=============')
            for index_c, item in row.items():
                if index_c == 'ID':
                    continue
                if pd.notnull(item):
                    type_str = str(type(item))
                    if 'time' in type_str:
                        pass
                    else:
                        print('Erasing:', item)
                        df_up.loc[index_r, index_c] = np.nan

    def relate_blood_draw_code_to_col_name(self):
        #This function uses the PARENT object.
        #Generate a dictionary to relate the
        #blood draw code to the column name in the M file.
        for col_name in self.parent.df.columns:
            if col_name.lower().startswith(self.blood_draw):
                code = self.extract_letter_code(col_name)
                self.blood_draw_code_to_col_name[code] = col_name
                self.i_blood_draw_code_to_col_name[col_name] = code
        #print(self.blood_draw_code_to_col_name)



    def check_df_dates_using_SID(self, df):
        #This function was modified on Oct 31, 2022
        #Check the dates in the given data frame
        #using the SID file.
        #The object df is passed by reference and is
        #modified using the loc method.
        #Notation
        #doc: date of collection
        print('Checking the dates of the LSM file using the SID.')
        doc = 'Date Collected'
        missing_dates = []
        for index_lsm, row_lsm in df.iterrows():
            ID = row_lsm['ID']
            full_ID = row_lsm['Full ID']
            if pd.isnull(ID):
                print(f'{full_ID=}')
                print('The entry in the ID column of the LSM file is empty.')
                raise ValueError('Empty ID.')
            letter_code = self.extract_letter_code(full_ID)
            col_name_for_code = self.blood_draw_code_to_col_name[letter_code]
            #Note that the doc cannot be empty because
            #it is connected to the letter code portion
            #of the full id.
            doc_LSM = row_lsm[doc]
            selector = self.parent.df['ID'] == ID
            if not selector.any():
                print(f'{ID=}')
                raise ValueError('This ID DNE.')
            doc_M = self.parent.df.loc[selector, col_name_for_code]
            if doc_M.isnull().any():
                print(f'{full_ID=}')
                missing_dates.append(full_ID)
                continue
                #raise ValueError('The date for this ID DNE in the M file.')
            doc_M = doc_M.iloc[0]
            if pd.notnull(doc_LSM):
                delta = (doc_M - doc_LSM) / np.timedelta64(1, 'D')
                delta = np.abs(delta)
                if 1 < delta:
                    print(f'{full_ID=}')
                    print('M file  :', doc_M)
                    print('LSM file:', doc_LSM)
                    raise ValueError('The date for this ID does not match the M file.')
            else:
                #If the date is empty in the LSM file, we 
                #simply complete it with the M file.
                print(f'{full_ID=} DOC was assigned from SID.')
                df.loc[index_lsm, doc] = doc_M
        if  0 < len(missing_dates):
            print('The following Full IDs have missing dates:')
            S = pd.Series(missing_dates)
            print(S.to_string(index=False))
        else:
            print('All dates in the LSM file are consistent.')
        #Overwriting the LSM file should be done externally for
        #safety reasons.


    def how_many_samples(self):
        R = slice('Blood Draw:Baseline - B',
                'Blood Draw:Repeat - JR')
        counter = 0
        full_counter = 0
        for index, row in self.parent.df.iterrows():
            potential_dates=row[R]
            for item in potential_dates:
                print(item)
                full_counter += 1
                s = str(type(item))
                if 'time' in s:
                    print(s)
                    print('>>>>>>>>>>>Yes')
                    counter += 1
        print(f'The number of dates is: {counter}.')
        print(f'The number of cells is: {full_counter}.')
        print(f'The % of used cells is: {counter/full_counter*100}.')

    #Dec 09 2022
    def migrate_dates_from_SID_to_LSM(self):
        blood_cols = self.i_blood_draw_code_to_col_name.keys()
        sample_counter = 0
        processed_counter = 0
        unprocessed_counter = 0
        dc = {'Full ID':[], 'Date Collected':[]}
        for index_m, row_m in self.parent.df.iterrows():
            print('=====================')
            ID = row_m['ID']
            for col, code in self.i_blood_draw_code_to_col_name.items():
                data = row_m[col]
                if pd.isnull(data):
                    continue
                dtype = str(type(data))
                if 'time' in dtype:
                    sample_counter += 1
                    doc = data
                    full_ID = ID + '-' + code
                    print(full_ID)
                    selection = self.parent.LSM_obj.df['Full ID'] == full_ID
                    if selection.any():
                        processed_counter += 1
                        continue
                    #At this point we know the sample is not in the LSM
                    dc['Full ID'].append(full_ID)
                    dc['Date Collected'].append(doc)
                    unprocessed_counter += 1
        df_up = pd.DataFrame(dc)
        df_up['p Full ID'] = df_up['Full ID']
        print(df_up)

        status_pre = self.parent.MPD_obj.compute_data_density(self.parent.LSM_obj.df)

        self.parent.LSM_obj.direct_serology_update_with_headers(df_up)

        print(f'{sample_counter=}')
        print(f'{processed_counter=}')
        print(f'{unprocessed_counter=}')

        status_post = self.parent.MPD_obj.compute_data_density(self.parent.LSM_obj.df)

        self.parent.MPD_obj.monotonic_increment_check(status_pre, status_post)

    def find_repeated_dates_in_the_M_file(self):
        #March 27 2023
        blood_slice = slice('Blood Draw:NoVac1 - NV1', 'Blood Draw:Repeat - JR')
        flag = False
        for index, row in self.parent.df.iterrows():
            ID = row['ID']
            dates = row[blood_slice]
            if dates.count() == 0:
                continue
            vc = dates.value_counts()
            s  = vc.gt(1)
            if s.any():
                flag = True
                vc = vc[s]
                print(ID)
                print(vc)
                print('===========')
        if not flag:
            print('SAFE with dates in the Sample Inventory File.')
        else:
            raise ValueError('Repeated dates in the SID.')


    def find_repeated_dates_in_megans_file(self):
        folder = 'Megan_mar_03_2023'
        fname = 'sid_clean.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        if os.path.exists(fname):
            df_up = pd.read_excel(fname)
            for col in df_up.columns:
                if col == 'ID':
                    continue
                df_up[col] = pd.to_datetime(df_up[col],
                        dayfirst=True, infer_datetime_format=True)
        else:
            fname = 'sid.xlsx'
            fname = os.path.join(self.parent.requests_path, folder, fname)
            df_up = pd.read_excel(fname,
                    skiprows=[0,1,2],
                    usecols='A,W:BF',
                    sheet_name='All Sites - AutoFill')
            self.format_megans_update(df_up)
            fname = 'sid_clean.xlsx'
            fname = os.path.join(self.parent.requests_path, folder, fname)
            df_up.to_excel(fname, index=False)
            return

        #print(df_up)
        #self.parent.print_column_and_datatype(df_up)
        L = []
        for index_up, row_up in df_up.iterrows():
            ID = row_up['ID']
            dates = row_up.iloc[1:]
            if dates.count() == 0:
                continue
            vc = dates.value_counts()
            s  = vc.gt(1)
            if s.any():
                vc = vc[s]
                #print(vc)
                #print(ID)
                date = vc.index[0]
                date_str = date.strftime('%d-%b-%y')
                L.append((ID,date_str))
                print('==============')
        df = pd.DataFrame(L, columns=['ID','Date'])
        fname = 'repeated_dates.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df.to_excel(fname, index=False)

