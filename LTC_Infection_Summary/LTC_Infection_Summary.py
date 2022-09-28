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
# <b> Section: LTC_InfectionSummary </b>
# From this Excel file we have to extract the PCR and DBS infections by date.
# Note that the wave type can be computed separately
# and is not part of the raw data. Moreover, using the VLOOKUP
# function for this task is not appropriate since it fails when
# the given date is before any of the dates in the lookup table.

class LTCInfectionSummary:
    def __init__(self, path, df=None):

        self.df = None
        self.dpath = path


        self.merge_column = 'ID'
        wave_fname = "LTC_wave_start_dates.xlsx"
        self.wave_fname = os.path.join(self.dpath, wave_fname)

        self.df_waves   = None
        self.positive_date_cols  = []
        self.positive_type_cols  = []
        self.wave_of_inf_cols    = []
        self.max_n_of_infections = 0
        self.n_waves             = 0
        self.long_wave_to_short  = {}

        self.wave_type = 'Dominant Strain'
        self.wave_date = 'Wave Start Date'
        self.wave_short = 'Short'
        self.inf_sequence_to_count = {}
        self.list_of_inf_edges = []
        self.set_of_inf_labels = set()
        self.infection_label_to_index = {}

        self.outputs_path = os.path.join('..', 'outputs')

        self.M_pure_name = 'M'
        self.M_fname     = self.M_pure_name + '.xlsx'

        self.load_wave_types_and_dates()

        if df is not None:
            self.initialize_class_with_df(df)
            print('LIS class initialization from DF.')
        else:
            #Which row contains the first data entry in the Excel file
            self.excel_starts_at = 3
            LIS = 'LTC_Infection_Summary'
            self.fname = os.path.join(self.dpath, LIS + '.xlsx')
            #Read the Excel file containing the data
            self.df = pd.read_excel(self.fname,
                       sheet_name="Summary List-NEW",
                       usecols="A:Q", skiprows=[0])
            print('The LIS class has been loaded using local file.')


    def load_wave_types_and_dates(self):
        self.df_waves   = pd.read_excel(self.wave_fname, sheet_name="dates")
        print('Wave types and dates has been loaded inside the LIS class.')
        self.n_waves  = len(self.df_waves)
        for index, row in self.df_waves.iterrows():
            s_wave = row[self.wave_short]
            l_wave = row[self.wave_type]
            self.long_wave_to_short[l_wave] = s_wave


    def remove_site_column(self):
        #We drop the Site column if present
        if 'Site' in self.df.columns:
            self.df.drop(columns=['Site'], inplace=True)
            print('Site column has been removed.')

    def check_for_repeats(self):
        #We drop rows containing NA in the "ID" from the
        #file because sometimes there are additional comments
        #below the table that get interpreted as additional rows.
        self.df.dropna(subset=[self.merge_column], inplace=True)
        value_count_gt_1 = self.df[self.merge_column].value_counts().gt(1)
        if value_count_gt_1.any():
            print('We have repetitions in this data frame')
            print('Check the column:', col_name)
            raise ValueError('No repetitions were expected.')


    def extract_date_type_and_generate_headers(self):
    #Extract all positive date headers
    #and create the wave of infection list.
        rexp = re.compile('[0-9]+')
        for column in self.df:
            if column.startswith("Positive Date"):
                self.positive_date_cols.append(column)
                #We also generate the wave of infection header.
                s = 'Wave of Inf '
                obj = rexp.search(column)
                s += obj[0]
                self.wave_of_inf_cols.append(s)
            elif column.startswith("Positive Type"):
                self.positive_type_cols.append(column)
        self.max_n_of_infections = len(self.wave_of_inf_cols)




    def remove_old_wave_types(self):
        L = []
        for wave in self.wave_of_inf_cols:
            if wave in self.df.columns:
                L.append(wave)
        if len(L) == 0:
            return
        self.df.drop(columns=L, inplace=True)
        print('Wave infection types have been erased and will be recomputed.')


    def update_dates_of_infection(self):
        fname = "update_for_LTC_Infection_15_Sep_2022.xlsx"
        fname = os.path.join(self.dpath, fname)
        df_update = pd.read_excel(fname)
        df_update.dropna(axis=0, subset=self.merge_column, inplace=True)
        #List of columns that are NOT in the 
        #target data frame.
        L = []
        #All the columns of the df_update have to
        #be present in the target data frame.
        for col in df_update:
            if col not in self.df.columns:
                L.append(col)
        df_update.drop(columns=L, inplace=True)
        #Merging step
        M = pd.merge(self.df, df_update, on=self.merge_column, how='outer')
        for col in df_update.columns:
            if col == self.merge_column:
                continue
            left = col + '_x'
            right= col + '_y'
            M[col] = M[left].where(~M[left].isnull(), M[right])
            M.drop(columns=[left, right], inplace=True)
        dc = {}
        #The columns of the merged object have
        #to be sorted to have them alphabetically.
        original_cols = M.columns
        new_cols = original_cols.copy()
        #Note we are sorting an Index object.
        new_cols = new_cols.sort_values()
        #Assign the merged object 
        self.df = M[new_cols]
        print('Infection dates have been included.')


    def compute_waves_of_infection(self):
        for date_col_name, wave_col_name in zip(self.positive_date_cols,
                                                self.wave_of_inf_cols):
            #List of labels
            L = []
            for entry in self.df[date_col_name]:
                if pd.isnull(entry):
                    L.append(np.nan)
                    continue
                #Before first date
                if entry <= self.df_waves['Wave Start Date'].iloc[0]:
                    label = self.df_waves['Dominant Strain'].iloc[0]
                    L.append(label)
                    continue
                #After last date
                if self.df_waves['Wave Start Date'].iloc[-1] <= entry:
                    label = self.df_waves['Dominant Strain'].iloc[-1]
                    L.append(label)
                    continue
                #Iterate over the wave start dates.
                #We start from the 2nd date.
                for k in range(1,self.n_waves):
                    wave_date = self.df_waves['Wave Start Date'].iloc[k]
                    old_wave_date = self.df_waves['Wave Start Date'].iloc[k-1]

                    #Check if the entry is smaller than the given wave date.
                    #Note that we are starting to count from the 2nd wave.
                    #If the entry is less than that date, it formally 
                    #belongs to the previous wave. 
                    if entry < wave_date:
                        label = self.df_waves['Dominant Strain'].iloc[k-1]
                        L.append(label)
                        break
            self.df[wave_col_name] = L
        print('The infection waves have been updated.')


    def print_col_names_and_types(self):
        for name, dtype in zip(self.df.columns, self.df.dtypes):
            print(name, ':', dtype)


    def generate_excel_file(self):
        fname = 'LTC_Infection_Summary_X.xlsx'
        fname = os.path.join(self.dpath, fname)
        self.df.to_excel(fname, index=False)
        print('Excel file was produced.')

    def compare_data_frames(self):
        df1 = pd.read_excel('./LTC_Infection_Summary_X.xlsx')
        df2 = pd.read_excel('./LTC_Infection_Summary_Y.xlsx')
        if df1.equals(df2):
            print('They are equal')

    def load_main_frame(self):
        fname = 'LTC_Infection_Summary_X.xlsx'
        fname = os.path.join(self.dpath, fname)
        self.df = pd.read_excel(fname)
        print('Excel file was loaded.')

    ##########Sep 22 2022##################

    def load_the_M_file(self):
        #Load the merged file M
        print('Make sure that this is a clone of the encrypted version!')
        fname = self.M_fname
        fname = os.path.join(self.outputs_path, fname)
        self.df = pd.read_excel(fname)
        print('MPD_LIS_SID, aka the M file, has been loaded from Excel')
        print('Inside the LTC_Infection_Summary class.')

    def initialize_class_with_df(self, df):
        self.df = df
        self.extract_date_type_and_generate_headers()

    def update_the_dates_and_waves(self, df_up):
        print('Update the dates and waves')
        for index, row in df_up.iterrows():
            ID = row['ID']
            date = row['date']
            print(ID, date)
            selector = self.df['ID'] == ID
            for col in self.positive_date_cols:
                if self.df.loc[selector, col].isnull().all():
                    self.df.loc[selector, col] = date
                    print(col, self.df.loc[selector, col])
                    break
        print('The infection dates have been updated.')
        self.compute_waves_of_infection()


    def compute_all_infection_patterns(self):
        #TD: Please test this function because it was modified.
        self.load_the_M_file()
        self.extract_date_type_and_generate_headers()
        level = 0
        sequence = []
        count = 0
        previous_wave = ''
        self.recursive_infection_walker(self.df,
                                        level,
                                        sequence,
                                        count,
                                        previous_wave)


    def recursive_infection_walker(self,
                                   df,
                                   level,
                                   sequence,
                                   count,
                                   previous_wave):
        if len(df) == 0 or level == self.max_n_of_infections:
            t = tuple(sequence)
            self.inf_sequence_to_count[t] = count
        else:
            wave_col = self.wave_of_inf_cols[level]
            stagnant  = count
            for wave in self.df_waves[self.wave_type]:
                T = df[df[wave_col] == wave].copy()
                L = sequence.copy()
                n_rows = len(T)
                if 0 < n_rows:
                    stagnant -= n_rows
                    count = n_rows
                    L.append(wave)
                    if 0 < len(previous_wave):
                        #Create an edge
                        edge = (previous_wave,
                                level-1,
                                wave,
                                level,
                                count)
                        self.list_of_inf_edges.append(edge)
                    self.recursive_infection_walker(T,
                                                    level+1,
                                                    L,
                                                    count,
                                                    wave)
            if 0 < stagnant:
                    self.recursive_infection_walker([],
                                                    level+1,
                                                    sequence,
                                                    stagnant,
                                                    previous_wave)
                    if 1 <= level:
                        #Create an edge
                        edge = (previous_wave,
                                level-1,
                                previous_wave,
                                level-1,
                                stagnant)
                        self.list_of_inf_edges.append(edge)

    def write_sequence_of_infections_to_file(self):
        fname = 'all_infection_seq.csv'
        fname = os.path.join(self.outputs_path, fname)
        with open(fname, 'w') as f:
            for key, value in self.inf_sequence_to_count.items():
                cpact = [self.long_wave_to_short[x] for x in key]
                cpact = ''.join(cpact)
                txt = ','.join(key)
                pad = self.max_n_of_infections - len(key) + 1
                txt += ','*pad + cpact + ',' + str(value) + '\n'
                f.write(txt)
                print(key, value)
        print(f'Finished writing: {fname=}')

    def write_infection_edges_to_file(self):
        fname = 'list_of_edges.csv'
        fname = os.path.join(self.outputs_path, fname)
        with open(fname, 'w') as f:
            for edge in self.list_of_inf_edges:
                source, a, target, b, value= edge
                source_short = self.long_wave_to_short[source]
                target_short = self.long_wave_to_short[target]
                source = source_short + str(a+1)
                target = target_short + str(b+1)
                self.set_of_inf_labels.update((source, target))
                L = [source, target, str(value)]
                txt = ','.join(L)
                txt += '\n'
                f.write(txt)
        print(f'Finished writing: {fname=}')
        fname = 'labels_for_edges.csv'
        fname = os.path.join(self.outputs_path, fname)
        #Write list of labels and indices for edges
        with open(fname, 'w') as f:
            c_fun = lambda x: ord(x[0]) + 100*int(x[1])
            S = sorted(self.set_of_inf_labels, key = c_fun)
            for k,label in enumerate(S):
                f.write(label + ',')
                f.write(str(k) + ',')
                f.write('\n')
        print(f'Finished writing: {fname=}')

    def full_run(self):
        self.remove_site_column()
        self.check_for_repeats()
        self.extract_date_type_and_generate_headers()
        self.remove_old_wave_types()
        self.update_dates_of_infection()
        self.compute_waves_of_infection()
        self.generate_excel_file()
        #self.compare_data_frames()

        print('Module: LTC_infection_summary.py FINISHED')





