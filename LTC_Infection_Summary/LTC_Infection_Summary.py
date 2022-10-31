#JRR @ McMaster University
#Update: 10-Oct-2022
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
    def __init__(self, path, parent=None):
        #In this class we deal with the infections
        #and the vaccines.

        self.parent = None
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

        self.vaccine_type_cols  = []
        self.vaccine_date_cols  = []
        self.list_of_valid_vaccines = ['Pfizer',
                                       'Moderna',
                                       'AstraZeneca',
                                       'COVISHIELD',
                                       'BPfizerO',
                                       'BModernaO',]

        self.wave_type = 'Dominant Strain'
        self.wave_date = 'Wave Start Date'
        self.wave_short = 'Short'
        self.inf_sequence_to_count = {}
        self.list_of_inf_edges = []
        self.set_of_inf_labels = set()
        self.infection_label_to_index = {}

        self.pcr_true  = 'Had a PCR+'
        self.pcr_last  = 'Last PCR+'
        self.inf_last  = 'Last infection'
        self.inf_free  = 'Days since last inf'

        self.outputs_path = os.path.join('..', 'outputs')

        self.M_pure_name = 'M'
        self.M_fname     = self.M_pure_name + '.xlsx'

        self.load_wave_types_and_dates()

        if parent:
            self.parent = parent
            self.extract_inf_vac_headers()
            print('LIS class initialization from Manager.')
        else:
            raise ValueError('Parent object is unavailable.')


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
        if 'Site' in self.parent.df.columns:
            self.parent.df.drop(columns=['Site'], inplace=True)
            print('Site column has been removed.')

    def check_for_repeats(self):
        #We drop rows containing NA in the "ID" from the
        #file because sometimes there are additional comments
        #below the table that get interpreted as additional rows.
        self.parent.df.dropna(subset=[self.merge_column], inplace=True)
        value_count_gt_1 = self.parent.df[self.merge_column].value_counts().gt(1)
        if value_count_gt_1.any():
            print('We have repetitions in this data frame')
            print('Check the column:', col_name)
            raise ValueError('No repetitions were expected.')


    def extract_inf_vac_headers(self):
    #Extract all positive date headers
    #and create the wave of infection list.
        rexp = re.compile('[0-9]+')
        vaccine_type = 'vaccine type'
        vaccine_date = 'vaccine date'
        pos_date = 'positive date'
        pos_type = 'positive type'

        for column in self.parent.df:
            if column.lower().startswith(pos_date):
                self.positive_date_cols.append(column)
                #We also generate the wave of infection header.
                s = 'Wave of Inf '
                obj = rexp.search(column)
                s += obj[0]
                self.wave_of_inf_cols.append(s)
            elif column.lower().startswith(pos_type):
                self.positive_type_cols.append(column)
            elif vaccine_type in column.lower():
                self.vaccine_type_cols.append(column)
            elif vaccine_date in column.lower():
                self.vaccine_date_cols.append(column)
        self.max_n_of_infections = len(self.wave_of_inf_cols)




    def remove_old_wave_types(self):
        L = []
        for wave in self.wave_of_inf_cols:
            if wave in self.parent.df.columns:
                L.append(wave)
        if len(L) == 0:
            return
        self.parent.df.drop(columns=L, inplace=True)
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
            if col not in self.parent.df.columns:
                L.append(col)
        df_update.drop(columns=L, inplace=True)
        #Merging step
        M = pd.merge(self.parent.df, df_update, on=self.merge_column, how='outer')
        for col in df_update.columns:
            if col == self.merge_column:
                continue
            left = col + '_x'
            right= col + '_y'
            M[col] = M[left].where(M[left].notnull(), M[right])
            M.drop(columns=[left, right], inplace=True)
        dc = {}
        #The columns of the merged object have
        #to be sorted to have them alphabetically.
        original_cols = M.columns
        new_cols = original_cols.copy()
        #Note we are sorting an Index object.
        new_cols = new_cols.sort_values()
        #Assign the merged object 
        self.parent.df = M[new_cols]
        print('Infection dates have been included.')


    def compute_waves_of_infection(self):
        for date_col_name, wave_col_name in zip(self.positive_date_cols,
                                                self.wave_of_inf_cols):
            #List of labels
            L = []
            for entry in self.parent.df[date_col_name]:
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
            self.parent.df[wave_col_name] = L
        print('The infection waves have been updated.')


    def print_col_names_and_types(self):
        for name, dtype in zip(self.parent.df.columns,
                               self.parent.df.dtypes):
            print(name, ':', dtype)


    def generate_excel_file(self):
        fname = 'LTC_Infection_Summary_X.xlsx'
        fname = os.path.join(self.dpath, fname)
        self.parent.df.to_excel(fname, index=False)
        print('Excel file was produced.')

    def compare_data_frames(self):
        df1 = pd.read_excel('./LTC_Infection_Summary_X.xlsx')
        df2 = pd.read_excel('./LTC_Infection_Summary_Y.xlsx')
        if df1.equals(df2):
            print('They are equal')

    def load_main_frame(self):
        fname = 'LTC_Infection_Summary_X.xlsx'
        fname = os.path.join(self.dpath, fname)
        self.parent.df = pd.read_excel(fname)
        print('Excel file was loaded.')

    ##########Sep 22 2022##################

    def load_the_M_file(self):
        #Load the merged file M
        print('Make sure that this is a clone of the encrypted version!')
        fname = self.M_fname
        fname = os.path.join(self.outputs_path, fname)
        self.parent.df = pd.read_excel(fname)
        print('MPD_LIS_SID, aka the M file, has been loaded from Excel')
        print('Inside the LTC_Infection_Summary class.')

    def compute_all_infection_patterns(self):
        #TD: Please test this function because it was modified.
        #self.load_the_M_file()
        #self.extract_inf_vac_headers()
        level = 0
        sequence = []
        count = 0
        previous_wave = ''
        self.recursive_infection_walker(self.parent.df,
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
        self.extract_inf_vac_headers()
        self.remove_old_wave_types()
        self.update_dates_of_infection()
        self.compute_waves_of_infection()
        self.generate_excel_file()
        #self.compare_data_frames()

        print('Module: LTC_infection_summary.py FINISHED')


    def update_the_dates_and_waves(self, df_up):
        #Update on Oct 10, 2022
        #New infections can be included using this method.
        #The method of detection can also be included.
        #The DOB can also be included.
        #Check if the update DF has a column named "method".
        #MOD: Method of detection
        #DOB: Date of birth
        #DOI: Date of infection
        method_in_update = 'MOD' in df_up.columns
        dob_in_update    = 'DOB' in df_up.columns
        print('Update the dates and waves')
        #Iterate over the dates in the update.
        for index, row in df_up.iterrows():
            ID = row['ID']
            print(f'{ID=}')
            date = row['DOI']
            selector = self.parent.df['ID'] == ID
            if not selector.any():
                raise ValueError('ID does not exist.')
            if dob_in_update:
                dob_up= row['DOB']
                if self.parent.df.loc[selector,'DOB'].isnull().any():
                    self.parent.df.loc[selector,'DOB'] = dob_up
                else:
                    dob_m = self.parent.df.loc[selector,'DOB'].values[0]
                    delta = (dob_m - dob_up) / np.timedelta64(1,'D')
                    if np.abs(delta) < 10:
                        pass
                    elif np.abs(delta) < 700:
                        print(f'DOB {delta=}')
                        print(f'Replacing current DOB with update')
                        self.parent.df.loc[selector,'DOB'] = dob_up
                    else:
                        print(f'DOB {delta=}')
                        raise ValueError('Big discrepancy in DOB')
            flag_found_slot = False
            is_a_new_date     = False
            type_col_found  = ''
            #Iterate over the dates in the M file.
            for date_col, type_col in zip(self.positive_date_cols,
                                         self.positive_type_cols):
                if self.parent.df.loc[selector, date_col].notnull().all():
                    #There is already a date in this cell.
                    stored_date = self.parent.df.loc[selector, date_col].values[0]
                    delta = (stored_date - date) / np.timedelta64(1,'D')
                    print('DOI')
                    print('Time delta:', np.abs(delta), 'days.')
                    #X=https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9106377/
                    #Accoding to X, 20 days is a lower bound.
                    if np.abs(delta) < 10:
                        print('This DOI is compatible with the M dates.')
                        print('Delta < 10 ==> Probably not a new date.')
                        print('Moving to the next date.')
                        type_col_found = type_col
                        flag_found_slot = True
                        break
                    else:
                        #The dates are different, so keep looking.
                        continue
                else:
                    #This slot is available.
                    self.parent.df.loc[selector, date_col] = date
                    print(date_col, date)
                    type_col_found = type_col
                    flag_found_slot = True
                    is_a_new_date   = True
                    break
            if not flag_found_slot:
                txt = 'We need another column to include this new infection.'
                raise ValueError(txt)
            if method_in_update:
                method = row['MOD']
                self.parent.df.loc[selector, type_col_found] = method
                print(f'Method {method=} has been included.')
            else:
                pass
                #--------------------------------------Current
                #If empty, it will be filled later by the 
                #assume_PCR_if_empty() function
                #Otherwise, we leave it as it is.
                #--------------------------------------Before
                #Assume PCR: Instruction given by Tara
                #self.parent.df.loc[selector, type_col_found] = 'PCR'
                #print(f'Method PCR has been chosen.')
        print('The infection dates have been updated.')
        if method_in_update:
            print('The infection detection method has been updated.')
        self.compute_waves_of_infection()
        self.assume_PCR_if_empty()
        self.update_PCR_and_infection_status()

    def update_PCR_and_infection_status(self):
        #This function determines if there 
        #have been PCR confirmed infections.
        #It also computes the last infection
        #and the number of days since the last infection.
        print('Within LTC_Infection_Summary:')
        print('Updating PCR and infection status.')
        pcr_selector = self.parent.df['ID'].isnull()
        #inf_selector = self.parent.df['ID'].isnull()
        self.parent.df[self.pcr_last] = np.nan
        self.parent.df[self.inf_last] = np.nan
        pcr = 'PCR'
        for date_col, type_col in zip(self.positive_date_cols,
                                  self.positive_type_cols):
            pcr_loc_selector = self.parent.df[type_col] == pcr
            inf_loc_selector = self.parent.df[date_col].notnull()

            pcr_selector |= pcr_loc_selector
            #inf_selector |= inf_loc_selector

            self.parent.df[self.pcr_last] =\
                    self.parent.df[date_col].where(pcr_loc_selector,
                                                   self.parent.df[self.pcr_last])
            self.parent.df[self.inf_last] =\
                    self.parent.df[date_col].where(inf_loc_selector,
                                                   self.parent.df[self.inf_last])
        self.parent.df[self.pcr_true] = pcr_selector
        delta = pd.to_datetime('today').normalize() -\
                self.parent.df[self.inf_last]
        temp = delta / np.timedelta64(1, 'D')
        self.parent.df[self.inf_free] = temp


    def order_infections_and_vaccines(self):
        #October 8 2022
        print('Checking the order of the infections.')
        self.set_chronological_order(self.positive_date_cols,
                                     self.positive_type_cols,
                                     'Infections')
        print('Checking the order of the vaccines.')
        self.set_chronological_order(self.vaccine_date_cols,
                                     self.vaccine_type_cols,
                                     'Vaccines')


    def set_chronological_order(self,
                                col_set,
                                companion,
                                descriptor=''):
        #October 8 2022
        exp_indices = np.arange(len(col_set), dtype=int)
        for index, row in self.parent.df.iterrows():
            ID = row['ID']
            p_dates = row[col_set]
            p_types = row[companion]
            if p_dates.isnull().all():
                #print(f'{ID=} has no history for this information.')
                #print(descriptor)
                continue
            big_date= datetime.datetime(3000,1,1)
            clone   = p_dates.replace(np.nan, big_date)
            clone   = clone.values
            p_dates = p_dates.values
            p_types = p_types.values
            s_indices = np.argsort(clone)
            if not np.array_equal(s_indices, exp_indices):
                print(f'{ID=}: A permutation is necessary.')
                print(descriptor)
                print('Original:')
                print(p_dates)
                print(s_indices)
                p_dates = p_dates[s_indices]
                p_types = p_types[s_indices]
                self.parent.df.loc[index, col_set] = p_dates
                self.parent.df.loc[index, companion] = p_types
                print('Modified:')
                print(p_dates)
                #For testing purposes.
                raise ValueError('Permutation')

    def assume_PCR_if_empty(self):
        for index, row in self.parent.df.iterrows():
            ID = row['ID']
            p_dates = row[self.positive_date_cols]
            p_types = row[self.positive_type_cols]
            if p_dates.isnull().all():
                #No infection history available.
                continue
            p_dates = p_dates.values
            p_types = p_types.values
            for k, (p_date, p_type) in enumerate(zip(p_dates,
                                                     p_types)):
                if pd.notnull(p_date):
                    if pd.isnull(p_type):
                        col = self.positive_type_cols[k]
                        print(f'Setting PCR in {col=} for {ID=}')
                        self.parent.df.loc[index, col] = 'PCR'


    def check_vaccine_types(self):
        for index, row in self.parent.df.iterrows():
            ID = row['ID']
            v_types = row[self.vaccine_type_cols]
            if v_types.isnull().all():
                continue
            for k,vaccine in enumerate(v_types):
                if pd.isnull(vaccine):
                    continue
                else:
                    if vaccine not in self.list_of_valid_vaccines:
                        print(f'{vaccine=}')
                        vnew = vaccine.strip()
                        if vnew not in self.list_of_valid_vaccines:
                            raise ValueError('Invalid vaccine type')
                        else:
                            print(f'Using: {vnew=}')
                            col = self.vaccine_type_cols[k]
                            self.parent.df.loc[index, col] = vnew


    #Oct 19 2022
    def find_closest_date_to_from(self, target, series, when='before'):
        #The target is the given date to use as a reference.
        #The series contains the dates to choose from.
        #When indicates if we want a previous or posterior date.
        #Returns a tuple
        #(0) Series index.
        #(1) The date that was found.
        #(2) The distance in days between the two dates.
        #If we only have a date that took place 
        #exactly at the target date, we use it.

        delta = (series - target).dt.days
        if when == 'before':
            selector = delta.lt(0)
        elif when == 'after':
            selector = delta.gt(0)
        else:
            raise ValueError('Unexpected when.')

        if not selector.any():
            print(f'No dates {when} {target=}')
            selector = delta.eq(0)
            if not selector.any():
                print(f'No dates @ {target=}')
                return (np.nan, np.nan, np.nan)
            else:
                print(f'Warning: Only found date exactly @ {target=}')
        else:
            print(f'Found date {when} {target=}')

        delta = np.abs(delta.loc[selector])
        dates = series.loc[selector]
        int_index = delta.argmin()
        index = dates.index[int_index]
        return (index, dates.iloc[int_index], delta.iloc[int_index])

    def add_n_infections_column(self):
        #Count the number of infections for each individual.
        L = []
        for index, row in self.parent.df.iterrows():
            dates = row[self.positive_date_cols]
            n_infections = dates.count()
            L.append(n_infections)
        self.parent.df['# infections'] = L




    def get_serology_dates_for_infection_dates(self):
        #This function creates a column with all the infection dates.
        #It also includes the corresponding method of detection.
        self.parent.MPD_obj.add_site_column()
        self.add_n_infections_column()
        DOC = self.parent.LSM_obj.DOC
        type_cols     = self.positive_type_cols
        cols_to_melt  = self.positive_date_cols
        cols_to_melt += type_cols
        doe = self.parent.MPD_obj.DOE
        dor = self.parent.MPD_obj.DOR
        cols_to_keep  = ['ID', 'Active', doe, dor, 'Site', '# infections']
        df = self.parent.df.pivot_longer(index = cols_to_keep,
                column_names = cols_to_melt,
                names_to = ['Infection event', 'Infection type'],
                values_to = ['Infection date', 'Method'],
                names_pattern = ['Positive Date [0-9]+', 'Positive Type [0-9]+'],
                )
        df.dropna(subset=['Infection date'], axis=0, inplace=True)
        df.drop(columns=['Infection type'], inplace=True)
        df.sort_values(by=['ID','Infection event'], axis=0, inplace=True)
        print(df)
        #Add the new columns to the df
        states = ['before', 'after']
        Ig_cols = ['Nuc-IgG-100', 'Nuc-IgA-100']
        add_cols = ['Date', 'Days'] + Ig_cols
        new_col_names = []
        slicer = {'before':None, 'after':None}
        for state in states:
            #Specify that we are using the serology data
            L = ['S: ' + x + ' ' + state for x in add_cols]
            slicer[state] = slice(L[0], L[-1])
            new_col_names.extend(L)

        #Add new columns to the data frame.
        df = df.reindex(columns = df.columns.to_list() + new_col_names)
        #print(new_col_names)
        #print(slicer)
        #print(df)
        #Up to this point the code has been tested.
        #Time to call Serology.
        #Iterate over the rows of the infection data frame.
        for index, row in df.iterrows():
            ID = row['ID']
            i_date = row['Infection date']
            selector_lsm = self.parent.LSM_obj.df['ID'] == ID
            if not selector_lsm.any():
                print(f'{ID=} has no serology information.')
                continue
            print(f'Working with serology for {ID=}')
            dates_lsm = self.parent.LSM_obj.df.loc[selector_lsm,DOC]
            dc_lsm = {'before':[], 'after':[]}
            for state in states:
                (index_lsm,
                date_lsm,
                days_lsm) = self.find_closest_date_to_from(i_date,
                        dates_lsm, when=state)
                print(f'{date_lsm=}')
                if pd.isnull(date_lsm):
                    print(f'{state}: Date was not found')
                    continue
                Igs = self.parent.LSM_obj.df.loc[index_lsm, Ig_cols]
                dc_lsm[state].extend((date_lsm, days_lsm))
                dc_lsm[state].extend((Igs.values))
                df.loc[index, slicer[state]] = dc_lsm[state]


        fpure = 'infection_dates_delta.xlsx'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        df.to_excel(fname, index = False)
        print(f'The {fpure=} file has been written to Excel.')

    def compute_slopes_for_serology(self):
        fpure = 'infection_dates_delta.xlsx'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        df = pd.read_excel(fname)
        print(df)
        add_cols = ['Had 0 days?', 'Days elapsed',
                'Delta IgG', 'Delta IgA',
                'Slope IgG', 'Slope IgA',
                'Has slopes?',
                'Ratio IgG',
                'Ratio IgA']
        #Specify that we are working with serology data
        add_cols = ['S: ' + x for x in add_cols]
        states = ['before', 'after']
        Ig_cols = ['S: Nuc-IgG-100', 'S: Nuc-IgA-100']
        df = df.reindex(columns = df.columns.to_list() + add_cols)
        for index, row in df.iterrows():
            had_0_days = False
            days_count = 0
            seq_IgG  = np.full(2, np.nan)
            seq_IgA  = np.full(2, np.nan)
            for k,state in enumerate(states):
                days_col = 'S: Days ' + state
                if row[days_col] == 0:
                    had_0_days = True
                days_count += row[days_col]
                #IgG
                IgG_col = Ig_cols[0] + ' ' + state
                seq_IgG[k] = row[IgG_col]
                #IgA
                IgA_col = Ig_cols[1] + ' ' + state
                seq_IgA[k] = row[IgA_col]
            #Had 0 days?
            df.loc[index,'S: Had 0 days?'] = had_0_days
            #Days elapsed
            df.loc[index,'S: Days elapsed'] = days_count
            #Delta IgG
            delta_IgG = np.diff(seq_IgG)[0]
            df.loc[index,'S: Delta IgG'] = delta_IgG
            #Delta IgA
            delta_IgA = np.diff(seq_IgA)[0]
            df.loc[index,'S: Delta IgA'] = delta_IgA
            #Ratio IgG
            df.loc[index,'S: Ratio IgG'] = seq_IgG[1] / seq_IgG[0]
            #Ratio IgA
            df.loc[index,'S: Ratio IgA'] = seq_IgA[1] / seq_IgA[0]
            if days_count == 0:
                print('Slope cannot be computed')
                df.loc[index,'S: Has slopes?'] = False
                continue
            #Slope IgG
            mG = delta_IgG / days_count
            df.loc[index,'S: Slope IgG'] = mG
            #Slope IgA
            mA = delta_IgA / days_count
            df.loc[index,'S: Slope IgA'] = mA
            #Has slopes?
            has_slopes = pd.notnull(mG) & pd.notnull(mA)
            df.loc[index,'S: Has slopes?'] = has_slopes

        print(df)
        fpure = 'infection_dates_slope.xlsx'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        df.to_excel(fname, index = False)
        print(f'The {fpure=} file has been written to Excel.')








