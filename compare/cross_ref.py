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

    ##################Sep 27 2022

    def satisfy_consents_20_09_2022_request(self):
        fname = 'consents.xlsx'
        removal_states   = self.MPD_obj.removal_states
        removal_states_l = self.MPD_obj.removal_states_l
        COMM = self.MPD_obj.comments
        DOR  = self.MPD_obj.DOR
        DOE  = self.MPD_obj.DOE
        fname = os.path.join(self.requests_path, 'consents_SEP_20_2022', fname)
        df_up = pd.read_excel(fname)
        df_up.replace('?', np.nan, inplace=True)
        id_index  = self.df.columns.get_loc('ID')
        doe_index = self.df.columns.get_loc(self.MPD_obj.DOE)
        dor_index = self.df.columns.get_loc(self.MPD_obj.DOR)
        comments_index = self.df.columns.get_loc(self.MPD_obj.comments)
        reason_index = self.df.columns.get_loc(self.MPD_obj.reason)
        dne_counter = 0
        doe_update_counter = 0
        doe_add_counter = 0
        doe_discrepancy_counter = 0
        dor_update_counter = 0
        dor_add_counter = 0
        dor_discrepancy_counter = 0
        reason_update_counter = 0
        reason_discrepancy_counter = 0
        reason_add_counter = 0
        ncols = len(self.df.columns)
        for index, row in df_up.iterrows():
            ID  = row.ID
            comments = row.COMMENTS
            dor = row.DOR
            print(f'{ID=}')
            m_selector = self.df['ID'] == ID
            m_row = self.df.loc[m_selector]
            flag_dne = False
            if len(m_row) == 0:
                dne_counter += 1
                #R = np.empty((1,ncols), dtype=object)
                R = [None]*ncols
                print(f'{ID=} is NA.')
                print('Creating a new Row')
                R[id_index] = ID
                flag_dne = True
            elif 1 < len(m_row):
                raise ValueError('Duplicates')
            #Deal with the DOE
            if row[['D','M','Y']].notnull().all():
                day = str(int(row.D))
                month = str(int(row.M))
                year = str(int(row.Y))
                #doe = datetime.datetime(year, month, day)
                doe = year+'-'+month+'-'+day
                doe = pd.to_datetime(doe)
                if flag_dne:
                    R[doe_index] = doe
                else:
                    if m_row[DOE].notnull().all():
                        m_doe = m_row[DOE].values[0]
                        delta_doe_time = np.abs((m_doe - doe) / np.timedelta64(1,'D'))
                        if 1 < delta_doe_time:
                            print(f'{delta_doe_time=}')
                            doe_discrepancy_counter += 1
                        if 28 < delta_doe_time:
                            doe_update_counter += 1
                            self.df.loc[m_selector,DOE]  = doe
                    else:
                        self.df.loc[m_selector,DOE]  = doe
                        doe_add_counter += 1

            if pd.notnull(comments):
                print(f'{comments=}')
                c_low = comments.lower()
                found_a_reason = False
                #Deal with the Reasons
                for k,state_l in enumerate(removal_states_l):
                    if state_l in c_low:
                        state = removal_states[k]
                        found_a_reason = True
                        break
                if found_a_reason:
                    #If DNE then add the reason to the new row.
                    if flag_dne:
                        R[reason_index] = state
                    else:
                        if m_row['Reason'].notnull().all():
                            reason = m_row['Reason'].values[0]
                            if reason == state:
                                pass
                            else:
                                print('Discrepancy in Reason:')
                                print(f'{reason=} vs {state=}')
                                reason_discrepancy_counter += 1
                                if state == 'Deceased':
                                    self.df.loc[m_selector, 'Reason'] = state
                                    reason_update_counter += 1
                        else:
                            self.df.loc[m_selector, 'Reason'] = state
                            reason_add_counter += 1
                #Deal with the comments.
                if flag_dne:
                    R[comments_index] = comments
                else:
                    m_comments = ''
                    if m_row[COMM].notnull().all():
                        m_comments = m_row[COMM].values[0]
                        print(f'{m_comments=}')
                    full_comments = m_comments + '#' + comments + '.'
                    self.df.loc[m_selector,COMM] = full_comments
            #Deal with the DOR
            if pd.notnull(dor):
                dor = pd.to_datetime(dor)
                print(f'{dor=}')
                if flag_dne:
                    R[dor_index] = dor
                else:
                    if m_row[DOR].notnull().all():
                        m_dor = m_row[DOR].values[0]
                        print(f'{m_dor=}')
                        delta_dor_time = np.abs((m_dor - dor) / np.timedelta64(1,'D'))
                        if 1 < delta_dor_time:
                            print(f'{delta_dor_time=}')
                            dor_discrepancy_counter += 1
                        if 1 < delta_dor_time:
                            dor_update_counter += 1
                            self.df.loc[m_selector,DOR]  = dor
                    else:
                        self.df.loc[m_selector,DOR]  = dor
                        dor_add_counter += 1
            if flag_dne:
                #Add the new row.
                self.df.loc[len(self.df.index)] = R

        print("=======================")
        print(f'{dne_counter=}')
        print(f'{doe_discrepancy_counter=}')
        print(f'{doe_update_counter=}')
        print(f'{doe_add_counter=}')
        print(f'{dor_discrepancy_counter=}')
        print(f'{dor_update_counter=}')
        print(f'{dor_add_counter=}')
        print(f'{reason_update_counter=}')
        print(f'{reason_add_counter=}')

        #Update the status column
        self.MPD_obj.update_active_status_column()
        self.write_the_M_file_to_excel()



    def satisfy_rainbow_request(self):
        self.df['DOB'] = pd.to_datetime(self.df['DOB'])
        fname = 'rainbow.xlsx'
        default_reason = 'Withdraw from Study'
        default_comment = ''
        color_to_reason ={39:'Deceased'}
        color_to_comment={22:'No consent and remaining on original protocol.',
                          43:'Covid infection',
                          3:'Chart Access only',}
        forced_comment = '(code: Rainbow)'
        dor_calculated = 'DOR calculated.'
        site = 4
        fname = os.path.join(self.requests_path, 'rainbow', fname)
        protected = set()

        print('100000000000000000000000000')
        df_rain = pd.read_excel(fname, skiprows = [0],
                                sheet_name='4th dose protocol-2022')
        rx = re.compile('[0-9]+')
        for index, row in df_rain.iterrows():
            ID = self.from_site_and_txt_get_ID(site,row['ID'])
            print(f'{ID=}')
            m_selector = self.df['ID'] == ID
            m_row = self.df.loc[m_selector]

            if len(m_row) == 0:
                raise ValueError('ID not found')

            is_active = m_row['Active']
            print(f'{is_active=}')
            if ~is_active.all():
                raise ValueError('Should be active')
            else:
                protected.add(ID)

            d4_date = row['Fourth Dose']
            m_d4_date = m_row['Fourth Vaccine Date']
            #Try to convert the Rainbow cell into a date
            try:
                d4_date = pd.to_datetime(d4_date)
                #If the M cell is not empty, compare
                if m_d4_date.notnull().all():
                    delta_time = m_d4_date - d4_date
                    if 1 < np.abs(delta_time):
                        raise ValueError('Discrepancy in 4th dose date')
                else:
                    #Update
                    print(f'{d4_date=}')
                    print('4th dose date updated')
                    self.df.loc[m_selector,'Fourth Vaccine Date'] = d4_date
            except:
                pass

        df_rain = pd.read_excel(fname, skiprows = [0],
                                sheet_name='Original List')

        for index, row in df_rain.iterrows():
            print(index, '##########################')
            color = row['Color']
            purple = 39
            comment = ''
            ID = self.from_site_and_txt_get_ID(site,row['ID'])
            if ID in protected:
                print('Protected ID:', ID)
                continue
            m_selector = self.df['ID'] == ID
            m_row = self.df.loc[m_selector]
            if len(m_row) == 0:
                raise ValueError('ID not found')
            doe = m_row[self.MPD_obj.DOE].values[0]
            doe_plus_1 = doe + pd.offsets.DateOffset(years=1)
            lbd = self.SID_obj.get_last_blood_draw(ID)
            dob = row['DOB']
            print(f'{ID=}')
            print(f'{doe_plus_1=}')
            print(f'{color=}')
            print(f'{dob=}')
            date_limit = datetime.datetime(2022,9,1)

            #DOB for the Rainbow data
            if isinstance(dob, datetime.datetime):
                pass
            else:
                dob = pd.to_datetime(dob)
                print(f'***{dob=}')
            if m_row['DOB'].notnull().all():
                m_dob = pd.to_datetime(m_row['DOB'].values[0])
                #m_dob = m_row['DOB']
                print(f'{m_dob=}')
                time_delta = (m_dob - dob) / np.timedelta64(1,'D')
                if 1 < np.abs(time_delta):
                    raise ValueError('DOB discrepancy')
            else:
                print('No DOB in record.')
                #Update DOB
                self.df.loc[m_selector,'DOB'] = dob

            if lbd:
                print(f'{lbd=}')
                if lbd < doe_plus_1:
                    dor = doe_plus_1
                    if date_limit < dor:
                        print('Color:', color)
                        print('DOR changed to LBD:', date_limit)
                        dor = lbd
                else:
                    dor = lbd
            else:
                dor = doe_plus_1


            #If the date of removal is already present, keep it.
            if m_row[self.MPD_obj.DOR].notnull().all():
                print('DOR already present.')
                pass
            else:
                if color == purple:
                    #Deceased, but date is unknown.
                    pass
                else:
                    self.df.loc[m_selector,self.MPD_obj.DOR] = dor
                    comment += dor_calculated
            #If the reason is already present, keep it.
            if m_row[self.MPD_obj.reason].notnull().all():
                print('Reason already present.')
                pass
            else:
                reason = color_to_reason.get(color, default_reason)
                self.df.loc[m_selector,self.MPD_obj.reason] = reason
            #If comments are present, complement them.
            if m_row[self.MPD_obj.note].notnull().all():
                comment = m_row[self.MPD_obj.note].values[0]
                print(f'{comment=}')
            comment += color_to_comment.get(color, default_comment)
            comment += forced_comment
            self.df.loc[m_selector,self.MPD_obj.note] = comment

        #Update the status column
        self.MPD_obj.update_active_status_column()
        self.write_the_M_file_to_excel()

    def request_jbreznik_26_09_2022(self):
        #Sep 28, 2022
        #PCR + DBS + MPD
        fname = 'DBS.xlsx'
        folder = 'jbreznik_sep_26_2022'
        fname = os.path.join('..','requests',folder, fname)
        df_dbs = pd.read_excel(fname, sheet_name='Raw Data + Linking')
        #print(df_dbs)
        relevant_cols = ['ID',
                         'DBS: Visit',
                         'DBS: Date',
                         'DBS: Nuc IgG 2.5',
                         'DBS: Nuc BAU/ml IgG 0.15625',
                         'DBS: Nuc BAU/ml IgG 0.625',
                         'DBS: Nuc BAU/ml IgG 2.5']
        print(relevant_cols)
        df_dbs = df_dbs[relevant_cols]
        #Get rid of noncompliant rows based on their IDs.
        rexp_c = re.compile('[0-9]+[-][0-9]+')
        def is_id(txt):
            if txt is np.nan:
                return txt
            obj = rexp_c.search(txt)
            if obj:
                return obj.group(0)
            else:
                return np.nan
        df_dbs['ID'] = df_dbs['ID'].apply(is_id)
        df_dbs.dropna(subset='ID', axis=0, inplace=True)
        df_dbs['DBS: Date'] = pd.to_datetime(df_dbs['DBS: Date'])
        for col, dtype in zip(df_dbs.columns, df_dbs.dtypes):
            print(col, dtype)
        #print(df_dbs)
        #Relevant columns for the LSM
        relevant_cols = []
        for col in self.LSM_obj.df.columns:
            if col.startswith('Nuc-IgG') or col.startswith('Nuc-IgA'):
                relevant_cols.append(col)
        base = self.LSM_obj.df.columns[:3]
        relevant_cols = base.to_list() + relevant_cols
        print(relevant_cols)
        df_LSM = self.LSM_obj.df[relevant_cols]
        #Relevant columns for the M file
        pcr = 'PCR'
        pcr_true = 'PCR-confirmed infection'
        selector = self.df['ID'].isnull()
        for dt_col, dm_col in zip(self.LIS_obj.positive_date_cols,
                                  self.LIS_obj.positive_type_cols):
            t_selector = self.df[dm_col] == pcr
            self.df[dt_col] = self.df[dt_col].where(t_selector, np.nan)
            selector |= t_selector
        self.df[pcr_true] = selector
        relevant_cols = ['ID', pcr_true] + self.LIS_obj.positive_date_cols
        print(relevant_cols)
        df_M = self.df[relevant_cols]
        #####Time to merge
        m1 = pd.merge(df_dbs, df_LSM, on='ID', how='outer')
        m2 = pd.merge(m1, df_M, on='ID', how='outer')
        fname = 'DBS_SER_PCR.xlsx'
        fname = os.path.join(self.outputs_path, fname)
        m2.to_excel(fname, index=False)
        print(f'File {fname=} has been written to Excel.')


    def request_megan_28_09_2022(self):
        #Sep 29, 2022
        #This function is to update the MPD file.
        fname = 'req.xlsx'
        folder = 'Megan_S20_28_Sep_2022'
        fname = os.path.join('..','requests',folder, fname)
        df_up = pd.read_excel(fname)
        local_to_std = {'Pass':'Deceased', 'Dis':'Discharged'}
        letter_to_sex = {'F':'Female', 'M':'Male'}
        self.check_id_format(df_up, 'ID')
        def change_reason(txt):
            return local_to_std[txt]
        def expand_sex_letter(txt):
            return letter_to_sex[txt]
        self.print_column_and_datatype(df_up)
        df_up['Reason'] = df_up['Reason'].apply(change_reason)
        df_up['Sex'] = df_up['Sex'].apply(expand_sex_letter)
        DOR = 'Date Removed from Study'
        relevant_columns = ['ID', DOR, 'Sex', 'Reason']
        df_up = df_up[relevant_columns]
        print(df_up)
        M = self.merge_with_master(df_up, 'ID', kind='complement')
        #Ready to write.
        self.df = M
        self.write_the_M_file_to_excel()

    def update_master(self, df_m, kind='update+'):
        #This function used to be part of the SID class.
        #However, the Manager class has a function that does
        #the same. So we removed it to avoid redundancies.

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
            if kind == 'update++':
                #Only trust the update
                M[column] = M[right]
            elif kind == 'update+':
                #The update has a higher priority
                #Keep the update if not empty.
                #Otherwise, use the original.
                M[column] = M[right].where(M[right].notnull(), M[left])
            elif kind == 'original+':
                #Keep the original if not empty.
                #Otherwise, use the update.
                M[column] = M[left].where(M[left].notnull(), M[right])
            else:
                raise ValueError('Unexpected kind for the SID update')
            #Erase the left and right columns
            M.drop(columns = [left, right], inplace=True)

        #Use the original order of the columns
        return M[original_list_of_columns]




    def update_PCR(self):
        #Oct 03, 2022
        self.LIS_obj.update_PCR_and_infection_status()
        #self.write_the_M_file_to_excel()

    def neutra_data(self):
        #Oct 03, 2022
        self.LND_obj.clean_LND_file()
        self.print_column_and_datatype(self.LND_obj.df)
        #self.write_the_M_file_to_excel()
        self.write_df_to_excel(self.LND_obj.df)

    #Oct 10,11,12 2022
    def tara_req_oct_3_2022(self):
        fname = 'sv_data.xlsx'
        folder = 'Tara_oct_03_2022'
        fname = os.path.join('..','requests',folder, fname)
        df_up = pd.read_excel(fname, sheet_name='1')
        df_up.replace('n/a', np.nan, inplace=True)
        df_up.replace('refused', np.nan, inplace=True)
        df_up.replace('Refused', np.nan, inplace=True)
        df_up.replace('REFUSED', np.nan, inplace=True)
        df_up.replace('None', np.nan, inplace=True)
        df_up.replace('Unknown', np.nan, inplace=True)
        df_up.replace('Moved', 'Moved Out', inplace=True)
        df_up.replace(' ', np.nan, inplace=True)
        prev_inf_col = 'Date of Previous COVID Infection'

        self.print_column_and_datatype(df_up)
        print(df_up)
        clean_sp = lambda x: x.replace(' ','')
        df_up['ID'] = df_up['ID'].apply(clean_sp)
        self.check_id_format(df_up, 'ID')
        max_date = datetime.datetime(2000,1,1)
        print(max_date)
        relevant_cols = ['DOB'] + self.LIS_obj.vaccine_date_cols
        for col in relevant_cols:
            print(f'Working with {col=}')
            for index, date in df_up[col].items():
                if pd.notnull(date):
                    if isinstance(date, str):
                        date = pd.to_datetime(date,
                                              dayfirst=False,
                                              yearfirst=False)
                        df_up.loc[index, col] = date
                    if col == 'DOB' and isinstance(date, datetime.datetime):
                        if max_date < date:
                            date = date - pd.DateOffset(years=100)
                            df_up.loc[index, col] = date
                            print(date)
                    elif isinstance(date, datetime.datetime):
                        pass
                    else:
                        print(date)
                        raise ValueError('Unexpected type.')
            df_up[col] = pd.to_datetime(df_up[col])

        df_up[prev_inf_col] = df_up[prev_inf_col].astype('string')
        #print(df_up)
        self.print_column_and_datatype(df_up)
        #If the DOB in the MPD file is less than 700 days
        #from the date in the update, we'll use the update.
        #Otherwise, we keep the date from the MPD.
        #Sometime we have cells with two dates, e.g.,
        #date 1; date 2. Hence, we use a regexp to extract
        #those dates.
        rx = re.compile('[0-9][0-9a-zA-Z/-]{5,}')
        DNE = []
        for index, row in df_up.iterrows():
            #print('-------------')
            ID = row['ID']
            print(f'{ID=}')
            dob_up = row['DOB']
            #print(f'{dob_up=}')
            selector  = self.df['ID'] == ID
            if not selector.any():
                print('This ID does not exist in the M file.')
                DNE.append(ID)
                raise ValueError('DNE')
            dob_m = self.df.loc[selector,'DOB']
            if pd.notnull(dob_m).all():
                dob_m = dob_m.values[0]
                #print(f'{dob_m=}')
                delta = (dob_m - dob_up) / np.timedelta64(1,'D')
                delta = np.abs(delta)
                if delta < 700:
                    #print('Fine')
                    pass
                else:
                    #Since we plan to give priority to the
                    #update, if the difference is too large
                    #we empty the cell in the update so that
                    #we keep the value in the M file.
                    df_up.loc[index,'DOB'] = np.nan
            prev_inf = row[prev_inf_col]
            if pd.isnull(prev_inf):
                continue
            L = []
            date_obj = rx.findall(prev_inf)
            print(prev_inf)
            print(date_obj)
            for date in date_obj:
                print('Before transformation:')
                print(date)
                new_date = pd.to_datetime(date)
                L.append(new_date)
                print('After transformation:')
                print(new_date)
            #Get the vector of + infection dates from the M file.
            p_dates = self.df.loc[selector,
                                  self.LIS_obj.positive_date_cols]
            s = p_dates.notnull().values[0]
            p_dates = p_dates.values[0][s]
            #Iterate over the dates in the update file.
            for up_date in L:
                found = False
                k = -1
                #Iterate over the dates in the M file.
                for k,p_date in enumerate(p_dates):
                    #p_date = pd.to_datetime(p_date)
                    delta = (p_date - up_date) / np.timedelta64(1,'D')
                    delta = np.abs(delta)
                    if delta < 10:
                        found = True
                        break
                if not found:
                    print('This date was not found:')
                    print(up_date)
                    k += 1
                    p_date_col = self.LIS_obj.positive_date_cols[k]
                    print(f'We are going to include it at {p_date_col}.')
                    print(f'Note that the order might have to be corrected')
                    self.df.loc[selector, p_date_col] = up_date


        #Eliminate the previous infection column 
        #from the update since
        #the DF has already been modified.
        #Note that this works because we previously included
        #all the participants that were not present in the MPD.
        df_up.drop(columns=[prev_inf_col], inplace=True)
        self.df = self.merge_with_M_and_return_M(df_up, 'ID', kind='correct')

    def tara_req_oct_5_2022(self):
        fname = 'CovidLTC_DataCoordination_site_13_Oct_5_2022.xlsx'
        folder = 'Tara_oct_5_2022'
        fname = os.path.join('..','requests',folder, fname)
        df_up = pd.read_excel(fname, sheet_name='1', skiprows=[0,1])
        df_up.replace('DECLINED', np.nan, inplace=True)
        df_up.replace('n/a', np.nan, inplace=True)
        df_up.replace('refused', np.nan, inplace=True)
        df_up.replace('Refused', np.nan, inplace=True)
        df_up.replace('REFUSED', np.nan, inplace=True)
        df_up.replace('None', np.nan, inplace=True)
        df_up.replace('Unknown', np.nan, inplace=True)
        df_up.replace('Moved', 'Moved Out', inplace=True)
        df_up.replace('Spikevax bivalent', 'BModernaO', inplace=True)
        df_up['Sex'] = df_up['Sex'].replace('F','Female')
        df_up['Sex'] = df_up['Sex'].replace('M','Male')
        df_up.replace(' ', np.nan, inplace=True)
        df_up.replace(u'\xa0', np.nan, inplace=True)
        fun = lambda x: x.lower()
        for col in self.LIS_obj.vaccine_date_cols:
            df_up[col] = pd.to_datetime(df_up[col])
        self.print_column_and_datatype(df_up)
        self.df = self.merge_with_M_and_return_M(df_up, 'ID', kind='correct')
        self.write_the_M_file_to_excel()

    def missing_dates(self):
        #Oct 13, 2022
        fname = 'missing_dates_oct_12_2022.xlsx'
        folder = 'missing_dates_oct_12_2022'
        fname = os.path.join('..','requests',folder, fname)
        df = pd.read_excel(fname)
        for index, row in df.iterrows():
            find        = row['Original']
            replacement = row['New']
            selector = self.LSM_obj.df['Full ID'] == find
            if selector.any():
                print(f'Got {find=}')
                self.LSM_obj.df.loc[selector, 'Full ID'] = replacement
                print(f'Replaced it with {replacement=}')
        self.check_LSM_dates()
        selector = self.LSM_obj.df['Full ID'].value_counts().gt(1)
        if selector.any():
            selector = selector.loc[selector]
            print(selector)
        #self.LSM_obj.write_to_excel()


    #Oct 15 2022
    #obj.tara_req_oct_13_2022()
    #obj.extract_date_and_method()
    #obj.LIS_obj.order_infections_and_vaccines()
    #obj.write_the_M_file_to_excel()
    def extract_date_and_method(self):
        #This function was developed for the site_01 data
        #given by Tara on Oct 13 2022.
        #Example
        #ID              DOB      Date of infection
        #01-9416404  24/01/1933  1st March 2020 (PCR)
        fname = 'site_01.xlsx'
        folder = 'Tara_13_oct_2022'
        fname = os.path.join('..','requests',folder, fname)
        df_up = pd.read_excel(fname, sheet_name='Covid Infection data')
        method_rx = re.compile('[(][ ]*(?P<method>[a-zA-Z]+)[ ]*[)]')
        #Date of infection
        doi = []
        #Method of detection
        mod = []
        for index, txt in df_up['Date of infection'].items():
            method_obj = method_rx.search(txt)
            if method_obj:
                method = method_obj.group('method')
                match = method_obj.group(0)
                txt2 = txt.replace(match,'')
                if '/' in txt2:
                    date = pd.to_datetime(txt2, dayfirst=True)
                else:
                    date = pd.to_datetime(txt2)
                print(f'{date=}')
                doi.append(date)
                print(f'{method=}')
                mod.append(method)
            else:
                raise ValueError('Unable to parse string.')
            print('--------------')
        df_up['DOI'] = doi
        df_up['MOD'] = mod
        for index, txt in df_up['DOB'].items():
            if isinstance(txt, datetime.datetime):
                pass
            elif isinstance(txt, str):
                if '/' in txt:
                    date = pd.to_datetime(txt, dayfirst=True)
                    df_up.loc[index,'DOB'] = date
                else:
                    raise ValueError('Unexpected format')
            else:
                raise ValueError('Unexpected format')
        df_up['DOB'] = pd.to_datetime(df_up['DOB'])
        print(df_up)
        self.print_column_and_datatype(df_up)
        self.LIS_obj.update_the_dates_and_waves(df_up)

    def melt_date_faulty(self):
        #Oct 19, 2022
        #This does not preserve the correspondence 
        #between the Infection Date and Type.
        inf_dates = pd.melt(self.df,
                            id_vars=cols_to_keep,
                            value_vars=cols_to_melt)
        dc={'value':'Infection date', 'variable':'Infection event'}
        inf_dates.rename(columns=dc, inplace=True)
        #If there is no infection, we don't use this individual.
        inf_dates.dropna(subset=['Infection date'], axis=0, inplace=True)
        inf_dates['Method'] = np.nan
        #new_col = inf_dates['ID'].isnull()
        #new_col.replace(True, np.nan, inplace=True)
        for col in type_cols:
            inf_dates['Method'] =\
                    inf_dates['Method'].where(inf_dates['Method'].notnull(),
                                              inf_dates[col])

        inf_dates.drop(columns=type_cols, inplace=True)
        #We'll leave the sorting until the end.
        #inf_dates.sort_values(by='Infection date', axis=0, inplace=True)
        inf_dates.sort_values(by=['ID','Infection event'], axis=0, inplace=True)
        print(inf_dates)



    def tara_req_oct_13_2022(self):
        fname = 'site_01.xlsx'
        folder = 'Tara_13_oct_2022'
        fname = os.path.join('..','requests',folder, fname)
        df_up = pd.read_excel(fname, sheet_name='Participants')
        df_up.replace('DECLINED', np.nan, inplace=True)
        df_up.replace('n/a', np.nan, inplace=True)
        df_up.replace('N/A', np.nan, inplace=True)
        df_up.replace('refused', np.nan, inplace=True)
        df_up.replace('Refused', np.nan, inplace=True)
        df_up.replace('REFUSED', np.nan, inplace=True)
        df_up.replace('None', np.nan, inplace=True)
        df_up.replace('Unknown', np.nan, inplace=True)
        df_up.replace('Moved', 'Moved Out', inplace=True)
        df_up.replace('Spikevax bivalent', 'BModernaO', inplace=True)
        df_up.replace(' ', np.nan, inplace=True)
        df_up.replace(u'\xa0', np.nan, inplace=True)
        cols_with_dates = [self.MPD_obj.DOE]
        cols_with_dates += self.LIS_obj.vaccine_date_cols
        for col in cols_with_dates:
            for index, txt in df_up[col].items():
                if isinstance(txt, str):
                    if '-' in txt:
                        raise ValueError('Unexpected format for date string')
                    #Note that it is crucial to specify dayfirst
                    #for XX/XX/XXXX because it guarantees consistency.
                    date = pd.to_datetime(txt, dayfirst=True)
                    df_up.loc[index,col] = date
                elif pd.isnull(txt):
                    pass
                elif isinstance(txt, datetime.datetime):
                    pass
                else:
                    raise ValueError('Unexpected type.')
            #Guarantee that the column has the date format.
            df_up[col] = pd.to_datetime(df_up[col])
        print(df_up)
        self.print_column_and_datatype(df_up)
        self.df = self.merge_with_M_and_return_M(df_up, 'ID', kind='original+')






    #Oct-26-2022
    def check_zains(self):
        #This function was updated on 26-Oct-2022
        fpure = 'merged_file.xlsx'
        folder= 'Zain_24_oct_2022'
        fname = os.path.join(self.requests_path, folder, fpure)
        df    = pd.read_excel(fname)
        n_zains = len(df)
        dob_counter = 0
        sex_counter = 0
        vaccine_date_counter = 0
        vaccine_date_total = 0
        vaccine_type_counter = 0
        vaccine_type_total = 0
        vaccine_sessions = ['First', 'Second', 'Third', 'Fourth']
        new_participants = []
        n_cols = self.df.shape[1]
        fpure = 'zain_vs_merge_report_26_oct_2022.txt'
        fname = os.path.join(self.requests_path, folder, fpure)
        with open(fname, 'w') as f_report:
            for index, row in df.iterrows():
                flag_write_to_file = False
                ID = row['ID']
                print(f'{ID=}')
                selection = self.df['ID'] == ID
                df_s = self.df.loc[selection,:]
                if len(df_s) == 0:
                    print(f'{ID=} DNE in the M file.')
                    new_participants.append(ID)
                    dob_counter += 1
                    raise ValueError('Fixed on the M file.')
                #Due to the uniqueness of the ID,
                #the first index should be the only index.
                index_m = df_s.index[0]
                row_m = df_s.loc[index_m,:]
                #DOB
                dob_z = row['DOB']
                dob_m = row_m['DOB']
                if pd.isnull(dob_z):
                    pass
                else:
                    if pd.isnull(dob_m):
                        print('Updating dob in M file from Zains')
                        self.df.loc[index_m,'DOB'] = dob_z
                        dob_counter += 1
                    else:
                        delta = (dob_m - dob_z) / np.timedelta64(1,'D')
                        delta = np.abs(delta)
                        delta_years = delta / 365
                        if delta == 0:
                            #Match
                            pass
                        elif delta_years < 2:
                            print('DOB =/=, but less than 2 years.')
                            print(f'{delta_years=}')
                            print(f'{dob_z=}')
                            print(f'{dob_m=}')
                            print('Updating dob in M file from Zains')
                            self.df.loc[index_m,'DOB'] = dob_z
                            dob_counter += 1
                        else:
                            if not flag_write_to_file:
                                print(f'{ID=}',file=f_report)
                            flag_write_to_file = True
                            print('DOB =/=, more than 2 years')
                            print('DOB =/=, more than 2 years', file=f_report)
                            print(f'{delta_years=}')
                            print(f'{delta_years=}', file=f_report)
                            print(f'{dob_z=}')
                            print(f'{dob_z=}', file=f_report)
                            print(f'{dob_m=}')
                            print(f'{dob_m=}', file=f_report)
                            dob_counter += 1
                #Sex
                sex_z = row['Sex']
                sex_m = row_m['Sex']
                if pd.isnull(sex_z):
                    pass
                else:
                    if pd.isnull(sex_m):
                        print('Updating sex in M file from Zains')
                        self.df.loc[index_m,'Sex'] = sex_z
                        sex_counter += 1
                    else:
                        if sex_z != sex_m:
                            if not flag_write_to_file:
                                print(f'{ID=}',file=f_report)
                            flag_write_to_file = True
                            print('Sex =/=')
                            print('Sex =/=', file=f_report)
                            print(f'{sex_z=}')
                            print(f'{sex_z=}', file=f_report)
                            print(f'{sex_m=}')
                            print(f'{sex_m=}', file=f_report)
                            sex_counter += 1
                #Vaccine dates
                for session in vaccine_sessions:
                    flag_session_date_updated = False
                    s = session.lower()
                    month_txt = 'vacc_'+ s +'_month'
                    year_txt  = 'vacc_'+ s +'_year'
                    month_z = row[month_txt]
                    year_z = row[year_txt]
                    if pd.isnull(month_z) or pd.isnull(year_z):
                        pass
                    else:
                        month_z = int(month_z)
                        year_z = int(year_z)
                        vaccine_date_total += 1
                        date_txt = session + ' Vaccine Date'
                        v_date = row_m[date_txt]
                        if pd.isnull(v_date):
                            v_date_z  = datetime.datetime(year_z,month_z,1)
                            print(f'In {session=}:')
                            print('Updating vacc. date in M file from Zains')
                            self.df.loc[index_m,date_txt] = v_date_z
                            vaccine_date_counter += 1
                            flag_session_date_updated = True
                        else:
                            month_m = v_date.month
                            year_m  = v_date.year
                            delta_m = month_m != month_z
                            delta_y = year_m != year_z
                            if delta_m or delta_y:
                                if not flag_write_to_file:
                                    print(f'{ID=}',file=f_report)
                                flag_write_to_file = True
                                print(f'In {session=}:')
                                print(f'In {session=}:', file=f_report)
                                print('Vaccine date =/=')
                                print('Vaccine date =/=', file=f_report)
                                if delta_m:
                                    print(f'{month_z=}')
                                    print(f'{month_z=}', file=f_report)
                                    print(f'{month_m=}')
                                    print(f'{month_m=}', file=f_report)
                                if delta_y:
                                    print(f'{year_z=}')
                                    print(f'{year_z=}', file=f_report)
                                    print(f'{year_m=}')
                                    print(f'{year_m=}', file=f_report)
                                vaccine_date_counter += 1
                    #Vaccine type
                    type_txt = session + ' Vaccine Type'
                    type_z   = row[type_txt]
                    type_m   = row_m[type_txt]
                    if pd.isnull(type_z):
                        pass
                    else:
                        vaccine_type_total += 1
                        pfz = 'Pfizer'
                        if pfz in type_z:
                            type_z = pfz
                        if pd.isnull(type_m) and flag_session_date_updated:
                            print(f'In {session=}:')
                            print('Updating vacc. type in M file from Zains')
                            self.df.loc[index_m, type_txt] = type_z
                            vaccine_type_counter += 1
                        else:
                            if type_z != type_m:
                                if not flag_write_to_file:
                                    print(f'{ID=}',file=f_report)
                                flag_write_to_file = True
                                print(f'In {session=}:')
                                print(f'In {session=}:', file=f_report)
                                print('Vaccine type =/=')
                                print('Vaccine type =/=', file=f_report)
                                print(f'{type_z=}')
                                print(f'{type_z=}', file=f_report)
                                print(f'{type_m=}')
                                print(f'{type_m=}', file=f_report)
                                vaccine_type_counter += 1

                if flag_write_to_file:
                    print('\n', file=f_report)




        sex_percent = sex_counter / n_zains * 100
        dob_percent = dob_counter / n_zains * 100
        vaccine_date_percent = vaccine_date_counter / vaccine_date_total * 100
        vaccine_type_percent = vaccine_type_counter / vaccine_type_total * 100


        print(f'{n_zains=}')
        print(f'{dob_counter=}')
        print(f'{dob_percent=}')
        print(f'{sex_counter=}')
        print(f'{sex_percent=}')
        print(f'{vaccine_date_counter=}')
        print(f'{vaccine_date_percent=}')
        print(f'{vaccine_type_counter=}')
        print(f'{vaccine_type_percent=}')
        print(f'{new_participants=}')




#obj = Comparator()
#obj.load_the_rainbow()

#obj.tara_req_oct_3_2022()
#obj.LIS_obj.order_infections_and_vaccines()
#obj.LIS_obj.assume_PCR_if_empty()
#obj.LIS_obj.update_PCR_and_infection_status()
#obj.MPD_obj.update_active_status_column()
#obj.write_the_M_file_to_excel()

#Requests from inception
#obj.full_run()
#obj.load_components_MPD_LIS_SID()
#obj.merge_MPD_LIS_SID_components()
#obj.load_the_M_file()
#obj.satisfy_request()
#obj.compute_all_infection_patterns()
#obj.write_sequence_of_infections_to_file()
#obj.write_infection_edges_to_file()
#obj.update_master_using_SID()
#obj.check_id_format()
#obj.extract_and_update_DOR_Reason_Infection()
#obj.merge_M_with_LSM()
#obj.satisfy_rainbow_request()
#obj.satisfy_consents_20_09_2022_request()
#obj.update_LSM()
#obj.request_jbreznik_26_09_2022()
#obj.check_id_format()
#obj.request_megan_28_09_2022()
#obj.extract_and_update_DOR_Reason_Infection()
#obj.extract_and_update_infection_detection_method()
#obj.write_the_M_file_to_excel()
#obj.update_LSM()
#obj.merge_M_with_LSM()
#obj.update_PCR()
#obj.neutra_data()
#obj.update_LSM()
#obj.check_LSM_dates()
#obj.update_LSM()
#obj.REP_obj.plot_n_infections_pie_chart()
#obj.LIS_obj.check_vaccine_types()
#obj.REP_obj.plot_vaccine_choice()
#obj.REP_obj.plot_infections_by_dates(30)
#obj.extract_and_update_DOR_Reason_Infection()
#obj.tara_req_oct_5_2022()
#obj.merge_M_with_LSM()
