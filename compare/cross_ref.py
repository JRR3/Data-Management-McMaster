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
        #Too specific.
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
        #Too specific.
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
        #Too specific.
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

    def plot_serology_slopes_from_selection(self):
        #It also generates the histograms for IgG and IgA in
        #the same plot.
        fpure = 'infection_dates_slope.xlsx'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        df    = pd.read_excel(fname, sheet_name = 'selection')
        fig, ax = plt.subplots()
        n_rows = len(df)
        mk_size = 4
        for k, (index, row) in enumerate(df.iterrows()):
            di = row['Infection date']
            d1 = row['S: Date before']
            d2 = row['S: Date after']
            dt_1 = row['S: Days before']
            IgG1 = row['S: Nuc-IgG-100 before']
            IgG2 = row['S: Nuc-IgG-100 after']
            IgA1 = row['S: Nuc-IgA-100 before']
            IgA2 = row['S: Nuc-IgA-100 after']
            IgG_slope = row['S: Slope IgG']
            IgA_slope = row['S: Slope IgA']
            IgG_mid = IgG1 + dt_1 * IgG_slope
            IgA_mid = IgA1 + dt_1 * IgA_slope
            ax.plot([d1,d2], [IgG1, IgG2], 'b-')
            ax.plot([di], [IgG_mid], 'ko', markersize=mk_size)
            ax.plot([d1,d2], [IgA1, IgA2], 'r-')
            ax.plot([di], [IgA_mid], 'ko', markersize=mk_size)
            if k == n_rows - 1:
                ax.plot([d1,d2], [IgG1, IgG2], 'b-', label='IgG')
                ax.plot([d1,d2], [IgA1, IgA2], 'r-', label='IgA')
        plt.legend(loc='best')
        ax.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b-%y"))
        ax.set_ylabel('OD')
        plt.xticks(rotation=45)
        fpure = 'slope_plot_selection.png'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        fig.savefig(fname)

        plt.close('all')
        fig, ax = plt.subplots()
        G_h, G_bins = np.histogram(df['S: Slope IgG'])
        A_h, A_bins = np.histogram(df['S: Slope IgA'], bins=G_bins)
        G_width = (G_bins[1] - G_bins[0])
        A_width = (A_bins[1] - A_bins[0])
        width = np.max((G_width, A_width))
        left = np.min((G_bins[0], A_bins[0]))
        right = np.max((G_bins[-1], A_bins[-1]))
        if left < 0 and right > 0:
            n = int(np.ceil(np.abs(left) / width))
            left_partition = np.flip(-np.linspace(0,n*width,n+1))
            n = int(np.ceil((right) / width))
            right_partition = np.linspace(width,n*width,n+1)
            partition = np.concatenate((left_partition, right_partition))
        else:
            n_left = np.ceil(np.abs(left) / width)
            if right < 0:
                n_right = np.floor(np.abs(right) / width)
            else:
                n_right = np.ceil(np.abs(right) / width)
            a = np.sign(left)*n_left*width
            b = np.sign(right)*n_right*width
            partition = np.arange(a,b,width)

        width /= 3
        fs = 12
        G_h, _ = np.histogram(df['S: Slope IgG'], bins=partition)
        A_h, _ = np.histogram(df['S: Slope IgA'], bins=partition)
        #Generate labels
        base = partition[0]
        L = []
        for x in partition[1:]:
            #print(base, x)
            interval = '({:.1E},{:.1E})'.format(base, x)
            interval = interval.replace('+0','+')
            interval = interval.replace('-0','-')
            interval = interval.replace('-.0','0')
            interval = interval.replace('+.0','0')
            #print(interval)
            base = x
            L.append(interval)
            #print('---')
        ax.bar(L, G_h, width=0.5,  facecolor='blue', label='IgG')
        ax.bar(L, A_h,  width=0.25, facecolor='red', label='IgA')
        ax.set_xlabel('Slope')
        ax.set_ylabel('Count')
        #ticks = ax.xaxis.get_ticklocs()
        #print(ticks)
        #ticklabels = ax.xaxis.get_ticklabels()
        #print(ticklabels)
        #ax.xaxis.set_ticks(ticks)
        #ax.xaxis.set_ticklabels(L)
        plt.xticks(fontsize=fs, rotation=90)
        #plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        plt.legend(loc='best', fontsize=fs)
        plt.tight_layout()
        fpure = 'hist_plot_Igs.png'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        fig.savefig(fname)

    def plot_serology_one_Ig_from_selection(self):
        fpure = 'infection_dates_slope.xlsx'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        df    = pd.read_excel(fname, sheet_name = 'selection')
        figG, axG = plt.subplots()
        figA, axA = plt.subplots()
        n_rows = len(df)
        mk_size = 4
        for k, (index, row) in enumerate(df.iterrows()):
            di = row['Infection date']
            d1 = row['S: Date before']
            d2 = row['S: Date after']
            dt_1 = row['S: Days before']
            IgG1 = row['S: Nuc-IgG-100 before']
            IgG2 = row['S: Nuc-IgG-100 after']
            IgA1 = row['S: Nuc-IgA-100 before']
            IgA2 = row['S: Nuc-IgA-100 after']
            IgG_slope = row['S: Slope IgG']
            IgA_slope = row['S: Slope IgA']
            IgG_mid = IgG1 + dt_1 * IgG_slope
            IgA_mid = IgA1 + dt_1 * IgA_slope
            axG.plot([d1,d2], [IgG1, IgG2], 'b-')
            axG.plot([di], [IgG_mid], 'ko', markersize=mk_size)
            axA.plot([d1,d2], [IgA1, IgA2], 'r-')
            axA.plot([di], [IgA_mid], 'ko', markersize=mk_size)
            if k == n_rows - 1:
                axG.plot([d1,d2], [IgG1, IgG2], 'b-', label='IgG')
                axA.plot([d1,d2], [IgA1, IgA2], 'r-', label='IgA')
        axG.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=1))
        axG.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b-%y"))
        axG.set_ylabel('OD')
        plt.legend(loc='best')
        plt.xticks(rotation=45)
        folder1= 'Braeden_oct_20_2022'
        folder2= 'oct_31_2022'
        fpure = 'slope_G_1.png'
        fname = os.path.join(self.parent.requests_path,
                folder1,
                folder2,
                fpure)
        figG.savefig(fname)

        axA.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=1))
        axA.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b-%y"))
        axA.set_ylabel('OD')
        plt.legend(loc='best')
        plt.xticks(rotation=45)
        fpure = 'slope_A_1.png'
        fname = os.path.join(self.parent.requests_path,
                folder1,
                folder2,
                fpure)
        figA.savefig(fname)

        plt.close('all')
        fig = pxp.histogram(df, x='S: Ratio IgG', color_discrete_sequence=['darkblue'])
        fig.update_traces(xbins=dict( # bins used for histogram
                    start=1.4, end=22))
        #fig.update_traces(line_width=5)
        fig.update_layout(font_size=20)
        fig.update_layout(hoverlabel={'font_size':20})
        fpure = 'ratio_G_1.html'
        fname = os.path.join(self.parent.requests_path,
                folder1,
                folder2,
                fpure)
        fig.write_html(fname)

        plt.close('all')
        fig = pxp.histogram(df, x='S: Slope IgG')
        fig.update_traces(xbins=dict( # bins used for histogram
                    start=0.001, end=0.036))
        #fig.update_traces(line_width=5)
        fig.update_layout(font_size=20)
        fig.update_layout(hoverlabel={'font_size':20})
        fpure = 'slope_G_1.html'
        fname = os.path.join(self.parent.requests_path,
                folder1,
                folder2,
                fpure)
        fig.write_html(fname)


        plt.close('all')
        fig = pxp.histogram(df, x='S: Ratio IgA')
        fig.update_traces(xbins=dict( # bins used for histogram
                    start=3, end=9))
        #fig.update_traces(line_width=5)
        fig.update_layout(font_size=20)
        fig.update_layout(hoverlabel={'font_size':20})
        fpure = 'ratio_A_2.html'
        fname = os.path.join(self.parent.requests_path,
                folder1,
                folder2,
                fpure)
        fig.write_html(fname)

        plt.close('all')
        fig = pxp.histogram(df, x='S: Slope IgA')
        fig.update_traces(xbins=dict( # bins used for histogram
                    start=0.0025, end=0.02))
        #fig.update_traces(line_width=5)
        fig.update_layout(font_size=20)
        fig.update_layout(hoverlabel={'font_size':20})
        fpure = 'slope_A_2.html'
        fname = os.path.join(self.parent.requests_path,
                folder1,
                folder2,
                fpure)
        fig.write_html(fname)


    def plot_serology_slope_vs_days_after_infection(self):
        fpure = 'infection_dates_slope.xlsx'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        df    = pd.read_excel(fname)
        df['S: Nearest IgG'] = df['S: Nuc-IgG-100 before'].where(
            False,
            #df['S: Days before'] < df['S: Days after'],
            df['S: Nuc-IgG-100 after'])
        df['S: Nearest IgA'] = df['S: Nuc-IgA-100 before'].where(
            False,
            #df['S: Days before'] < df['S: Days after'],
            df['S: Nuc-IgA-100 after'])
        selection  = df['Method'] == 'PCR'
        selection &= df['# infections'] == 1
        print('Available data (negative slopes included):')
        print(selection.value_counts())
        df_s = df.loc[selection, :]

        #Scatter
        plt.close('all')
        fig, ax = plt.subplots()
        ax.scatter(df_s['S: Days after'],
                   df_s['S: Slope IgG'], color='blue')
        #Regression
        selection = df_s['S: Slope IgG'] > 0
        df_G = df_s.loc[selection, :]
        n_G = len(df_G)
        print(f'{n_G=}')
        x = df_G['S: Days after'].values
        y_G = df_G['S: Slope IgG'].values
        f_G = np.polyfit(x, np.log(y_G), 1)
        k_G = f_G[0]
        c_G = np.exp(f_G[1])
        l_G = -np.log(2) / k_G
        print(f'{k_G=:.2E}')
        print(f'{c_G=:.2E}')
        print(f'{l_G=:.2E}')
        xx = np.linspace(np.min(x), np.max(x), 100)
        yy_G = c_G * np.exp( k_G * xx)
        ax.plot(xx, yy_G,'k-')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_xlabel('Days after infection')
        ax.set_ylabel('IgG slope')
        fpure = 'Slope_IgG_vs_days_after.png'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        fig.savefig(fname)

        #Scatter
        plt.close('all')
        fig, ax = plt.subplots()
        ax.scatter(df_s['S: Days after'], df_s['S: Slope IgA'], color='red')
        #Regression
        selection = df_s['S: Slope IgA'] > 0
        df_A = df_s.loc[selection, :]
        n_A = len(df_A)
        print(f'{n_A=}')
        x = df_A['S: Days after'].values
        y_A = df_A['S: Slope IgA'].values
        f_A = np.polyfit(x, np.log(y_A), 1)
        k_A = f_A[0]
        c_A = np.exp(f_A[1])
        l_A = -np.log(2) / k_A
        print(f'{k_A=:.2E}')
        print(f'{c_A=:.2E}')
        print(f'{l_A=:.2E}')
        xx = np.linspace(np.min(x), np.max(x), 100)
        yy_A = c_A * np.exp( k_A * xx)
        ax.plot(xx, yy_A,'k-')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_xlabel('Days after infection')
        ax.set_ylabel('IgA slope')
        fpure = 'Slope_IgA_vs_days_after.png'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        fig.savefig(fname)


        #=========================Nearest
        print('Nearest')
        #Scatter
        plt.close('all')
        fig, ax = plt.subplots()
        ax.scatter(df_s['S: Days after'],
                   df_s['S: Nearest IgG'], color='blue')
        #Regression
        selection = df_s['S: Nearest IgG'] > 0
        df_G = df_s.loc[selection, :]
        n_G = len(df_G)
        print(f'{n_G=}')
        x = df_G['S: Days after'].values
        y_G = df_G['S: Nearest IgG'].values
        f_G = np.polyfit(x, np.log(y_G), 1)
        k_G = f_G[0]
        c_G = np.exp(f_G[1])
        l_G = -np.log(2) / k_G
        print(f'{k_G=:.2E}')
        print(f'{c_G=:.2E}')
        print(f'{l_G=:.2E}')
        xx = np.linspace(np.min(x), np.max(x), 100)
        yy_G = c_G * np.exp( k_G * xx)
        ax.plot(xx, yy_G,'k-')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_xlabel('Days after infection')
        ax.set_ylabel('Nearest IgG')
        fpure = 'Nearest_IgG_vs_days_after.png'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        fig.savefig(fname)

        #Scatter
        plt.close('all')
        fig, ax = plt.subplots()
        ax.scatter(df_s['S: Days after'], df_s['S: Nearest IgA'], color='red')
        #Regression
        selection = df_s['S: Nearest IgA'] > 0
        df_A = df_s.loc[selection, :]
        n_A = len(df_A)
        print(f'{n_A=}')
        x = df_A['S: Days after'].values
        y_A = df_A['S: Nearest IgA'].values
        f_A = np.polyfit(x, np.log(y_A), 1)
        k_A = f_A[0]
        c_A = np.exp(f_A[1])
        l_A = -np.log(2) / k_A
        print(f'{k_A=:.2E}')
        print(f'{c_A=:.2E}')
        print(f'{l_A=:.2E}')
        xx = np.linspace(np.min(x), np.max(x), 100)
        yy_A = c_A * np.exp( k_A * xx)
        ax.plot(xx, yy_A,'k-')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_xlabel('Days after infection')
        ax.set_ylabel('Nearest IgA')
        fpure = 'Nearest_IgA_vs_days_after.png'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        fig.savefig(fname)

    def boxplot_serology_slope_vs_days_after_infection(self):
        fpure = 'infection_dates_slope.xlsx'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        df    = pd.read_excel(fname)
        df['S: Nearest IgG'] = df['S: Nuc-IgG-100 before'].where(
            df['S: Days before'] < df['S: Days after'],
            df['S: Nuc-IgG-100 after'])
        df['S: Nearest IgA'] = df['S: Nuc-IgA-100 before'].where(
            df['S: Days before'] < df['S: Days after'],
            df['S: Nuc-IgA-100 after'])
        selection  = df['Method'] == 'PCR'
        selection &= df['# infections'] == 1
        selection &= df['S: Has slopes?'] == True
        print('Available data (negative slopes included):')
        print(selection.value_counts())
        df_s = df.loc[selection, :]
        #=====Bins IgG
        plt.close('all')
        fig, ax = plt.subplots()
        t_max = df_s['S: Days after'].max()
        print(f'{t_max=}')
        width = 30
        n_bins = int(np.ceil(t_max / width))
        print(f'{n_bins=}')
        dc = {}
        base = 0
        bin_labels = []
        index_to_label = {}
        for k in range(1,n_bins+1):
            txt = f'[{base},{width*k})'
            bin_labels.append(txt)
            base = width*k
            index_to_label[k-1] = txt
            dc[txt] = []
            print(txt)
        for index, row in df_s.iterrows():
            days_after = row['S: Days after']
            IgG_slope = row['S: Slope IgG']
            bin_index = days_after // width
            bin_label = index_to_label[bin_index]
            dc[bin_label].append(IgG_slope)
        ax.boxplot(dc.values())
        ax.set_xticklabels(dc.keys())
        ax.set_ylabel('IgG (Slope)')
        ax.set_xlabel('Days after infection')
        plt.xticks(rotation=60)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.tight_layout()

        fpure = 'Slope_IgG_vs_bin_days_after.png'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        fig.savefig(fname)

        #=====Bins IgA
        plt.close('all')
        fig, ax = plt.subplots()
        t_max = df_s['S: Days after'].max()
        print(f'{t_max=}')
        width = 30
        n_bins = int(np.ceil(t_max / width))
        print(f'{n_bins=}')
        dc = {}
        base = 0
        bin_labels = []
        index_to_label = {}
        for k in range(1,n_bins+1):
            txt = f'[{base},{width*k})'
            bin_labels.append(txt)
            base = width*k
            index_to_label[k-1] = txt
            dc[txt] = []
            print(txt)
        for index, row in df_s.iterrows():
            days_after = row['S: Days after']
            IgA_slope = row['S: Slope IgA']
            bin_index = days_after // width
            bin_label = index_to_label[bin_index]
            dc[bin_label].append(IgA_slope)
        ax.boxplot(dc.values())
        ax.set_xticklabels(dc.keys())
        ax.set_ylabel('IgA (Slope)')
        ax.set_xlabel('Days after infection')
        plt.xticks(rotation=60)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.tight_layout()

        fpure = 'Slope_IgA_vs_bin_days_after.png'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        fig.savefig(fname)

    def find_short_jumps(self):
        #This function was updated on 13-Oct-2022
        #The purpose of this function is to find which
        #treatments are comparable. By comparable, we
        #mean that their difference has an l1-norm of 1.
        #Dr. Bowdish and Jenna decided not to pursue this direction.
        folder = 'Jenna_oct_6_2022'
        fname = 'table.xlsx'
        fname = os.path.join(self.requests_path, folder, fname)
        df = pd.read_excel(fname)
        print('# rows df:', len(df))
        rel_columns = ['Anti.inflamm.and.DMARDs',
                       'TNF.inhib',
                       'IL6.inhib',
                       'JAK.inhib',
                       'Costim.Inhib']
        df_med = df[rel_columns].copy()
        M = df_med.values
        M = M.astype(int)
        vec_to_count = {}
        #We are going to count how many unique rows we have.
        for row in M:
            s = str(row)
            v = vec_to_count.get(s,0)
            vec_to_count[s] = v + 1
        df_unique_rows = pd.Series(vec_to_count).to_frame()
        df_unique_rows = df_unique_rows.reset_index()
        df_unique_rows = df_unique_rows.rename(columns={'index':'Combination',
                                                        0:'Counts'})
        index_to_label = {k:chr(x) for k,x in enumerate(range(65,91))}
        vec_to_label = {}
        for k,(key,count) in enumerate(vec_to_count.items()):
            vec_to_label[key] = index_to_label[k]
        df_unique_rows = df_unique_rows.rename(index=index_to_label)

        print(df_unique_rows)
        n_unique_rows = len(vec_to_count)

        df_unique_rows = df_med.drop_duplicates()
        M = df_unique_rows.values

        G = nx.Graph()

        for k, row1 in enumerate(M):
            for j in range(k, n_unique_rows):
                row2 = M[j]
                dist = np.linalg.norm(row1-row2, 1)
                if dist < 1.5:
                    #Connected
                    s1 = str(row1)
                    s2 = str(row2)
                    l1 = index_to_label[k]
                    w1 = vec_to_count[s1]
                    l2 = index_to_label[j]
                    w2 = vec_to_count[s2]
                    if k == j:
                        continue
                        G.add_edge(l1,l2)
                    else:
                        l1 += ',' + str(w1)
                        l2 += ',' + str(w2)
                        G.add_edge(l1,l2)
                        G.add_edge(l1,l2)
        print(G)
        pos = nx.spring_layout(G)
        pos = nx.circular_layout(G)
        nx.draw_networkx_nodes(G,
                               pos,
                               cmap=plt.get_cmap('Wistia'),
                               node_color = list(vec_to_count.values()),
                               node_size = 800)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edge_color='b')
        #edge_labels=dict([((u,v),w) for u,v,w in G.edges.data('weight')])
        #nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
        plot_name = 'graph.png'
        fname = os.path.join(self.dpath, plot_name)
        plt.savefig(fname)
        #NOTE: Function is terminated abruptly.
        return
        unique_rows = df_med.drop_duplicates()
        print(unique_rows)
        print('# unique rows:', len(unique_rows))
        print('# unique rows:', n_unique_rows)
        return
        print(unique_rows.to_string(index=False))
        individuals = df['Individual'].unique()
        print('# of individuals:', df['Individual'].nunique())
        for x in individuals:
            selector = df['Individual'] == x
            rows = df.loc[selector, rel_columns]
            n_rows = len(rows)
            for med in rel_columns:
                if rows[med].nunique() == 1:
                    pass
                else:
                    print('==========')
                    print(x)
                    print(rows)

    #Oct 31 2022
    def whole_blood_update(self):
        folder = 'Megan_oct_31_2022'
        fname = 'whole_blood.xlsx'
        fname = os.path.join('..','requests',folder, fname)
        df_up = pd.read_excel(fname)
        print(df_up)
        self.df = self.merge_X_with_Y_and_return_Z(self.df,
                                         df_up,
                                         self.merge_column,
                                         kind='update+')

    def jessica_oct_31_2022(self):
        #Updating the neutralization data file.
        fname = 'mnt_data_updated.xlsx'
        folder = 'Jessica_oct_31_2022'
        fname = os.path.join('..','requests',folder, fname)
        df_up = pd.read_excel(fname)
        df_up.dropna(axis=0, subset='Full ID', inplace=True)
        if df_up[self.merge_source].value_counts().gt(1).any():
            selector = df_up[self.merge_source].value_counts().gt(1)
            duplicates = selector.loc[selector]
            print(duplicates)
            raise ValueError('There are duplicates in the update.')
        fname = 'missing_dates_oct_12_2022.xlsx'
        folder = 'Jessica_oct_31_2022'
        fname = os.path.join('..','requests',folder, fname)
        df_change_id = pd.read_excel(fname)
        for index, row in df_change_id.iterrows():
            old_id = row['Original']
            new_id = row['New']
            selector = df_up['Full ID'] == old_id
            if selector.any():
                df_up.loc[selector, 'Full ID'] = new_id
        print(df_up)
        M = self.merge_X_with_Y_and_return_Z(self.LND_obj.df,
                                             df_up,
                                             self.merge_source,
                                             kind='update+')
        M = self.create_df_with_ID_from_full_ID(M)
        self.SID_obj.check_df_dates_using_SID(M)
        #self.write_df_to_excel(M, label='LND')
        M = self.merge_X_with_Y_and_return_Z(self.LSM_obj.df,
                                             M,
                                             self.merge_source,
                                             kind='update+')
        self.SID_obj.check_df_dates_using_SID(M)
        self.LSM_obj.df = M

    def tara_oct_31_2022(self):
        self.extract_and_update_DOR_Reason_Infection()
        self.df['Reason'] = self.df['Reason'].str.replace('Moved Out', 'Moved')
        print(self.df['Reason'].value_counts())

    def create_ahmad_infection_file(self):
        #Create the 1st version of Ahmad's file.
        #Nov 02 2022
        folder = 'Ahmad_oct_31_2022'
        fname = 'ahmad.xlsx'
        fname = os.path.join(self.requests_path, folder, fname)
        df_up = pd.read_excel(fname)
        df_up.replace('.', np.nan, inplace=True)
        self.print_column_and_datatype(df_up)
        L = ['Positive_date_1','Positive_type_1',
                'Positive_date_2','Positive_type_2',
                'Positive_date_3','Positive_type_3',
                'Positive_date_4','Positive_type_4',
                'Positive_date_5','Positive_type_5']
        def change_to_local(txt):
            txt = txt.replace('_date_',' Date ')
            txt = txt.replace('_type_',' Type ')
            return txt
        dc = {}
        for label in L:
            dc[label] = change_to_local(label)
            if 'date' in label:
                df_up[label] = pd.to_datetime(df_up[label])
        df_up.rename(columns=dc, inplace=True)
        print(df_up)
        #self.write_df_to_excel(df_up,
                               #label='ahmads_infection_file')



    def compare_ahmad_infection_file_w_M(self):
        #November 03 2022
        #What is different between Ahmad's file
        #and the master file.
        self.LIS_obj.load_ahmad_file()

        for index, row in self.LIS_obj.df_ah.iterrows():
            ID = row['ID']
            selection = self.df['ID'] == ID
            print('------------------------------')
            print(f'{ID=}')
            print('------------------------------')
            if ~selection.any():
                raise ValueError('ID DNE')
            row_m = self.df[selection].iloc[0]
            for type_col, date_col in zip(self.LIS_obj.positive_type_cols,
                    self.LIS_obj.positive_date_cols):
                print('>>>>>>>>>')
                print(f'Looking at {date_col=}')
                print('>>>>>>>>>')
                d_a = row[date_col]
                t_a = row[type_col]
                d_m = row_m[date_col]
                t_m = row_m[type_col]
                if pd.isnull(d_a) and pd.isnull(d_m):
                    print('Both dates are empty')
                elif pd.isnull(d_a):
                    print('Ahmad date is null')
                    print(f'{d_m=}')
                    print(f'{t_m=}')
                elif pd.isnull(d_m):
                    print('Master date is null')
                    print(f'{d_a=}')
                    print(f'{t_a=}')
                else:
                    if t_m == t_a:
                        print('Method of detection is equal')
                    else:
                        print('Method of detection is not equal')
                        print(f'{t_m=}')
                        print(f'{t_a=}')
                    delta = d_a - d_m
                    delta /= np.timedelta64(1,'D')
                    delta = np.abs(delta)
                    if delta < 1:
                        print('Dates are equal')
                    else:
                        print('Dates are not equal')
                        print(f'{d_a=}')
                        print(f'{d_m=}')

    def tara_nov_04_2022(self):
        fname  = 'inf_and_removal_update.xlsx'
        folder = 'Tara_nov_04_2022'
        df_up = self.load_single_column_df_for_update(fname, folder)
        self.extract_and_update_DOR_Reason_Infection(df_up)



    def tara_nov_07_2022(self):
        fname  = 'vaccine_update.xlsx'
        folder = 'Tara_nov_07_2022'
        fname = os.path.join('..','requests',folder, fname)
        linf = 'last infection'
        type_dict = {'ID1': 'str',
                'ID2': 'str',
                linf: 'str',
                'DOB': 'str'}
        df_up = pd.read_excel(fname, dtype=str)
        df_up.replace('n/a', np.nan, inplace=True)
        df_up.replace('refused', np.nan, inplace=True)
        df_up.replace('Refused', np.nan, inplace=True)
        df_up.replace('REFUSED', np.nan, inplace=True)
        df_up.replace('None', np.nan, inplace=True)
        df_up.replace('Unknown', np.nan, inplace=True)
        df_up.replace('Yes - date unknown', np.nan, inplace=True)
        df_up.replace('COVISHEILD', 'COVISHIELD', inplace=True)
        df_up.replace(' ', np.nan, inplace=True)
        df_up['ID'] = ''
        #print(df_up)
        DOR=self.MPD_obj.DOR
        self.print_column_and_datatype(df_up)
        id_regexp = re.compile('(?P<site>[0-9]{2})[-]?(?P<code>[0-9]{4}[ ]?[0-9]{3})')
        def extract_id(txt):
            obj = id_regexp.search(txt)
            if obj:
                site = obj.group('site')
                code = obj.group('code')
                code = code.replace(' ','')
                ID   = site + '-' + code
                return ID
            else:
                return None

        yearfirst_regexp = re.compile('[0-9]{4}[-][0-9]{2}[-][0-9]{2}')
        dayfirst_regexp = re.compile('[0-9]+[-][a-zA-Z]+[-](?P<year>[0-9]+)')
        monthfirst_regexp = re.compile('(?P<month>[0-9]+)[/](?P<day>[0-9]+)[/](?P<year>[0-9]+)')
        def convert_str_to_date(txt):
            if pd.isnull(txt):
                raise ValueError('Object is NAN')
            if '/' in txt:
                #Month/Day/Year
                obj = monthfirst_regexp.search(txt)
                if obj:
                    date = obj.group(0)
                    month_str = obj.group('month')
                    month_int = int(month_str)
                    if 12 < month_int:
                        date = pd.to_datetime(date, dayfirst=True)
                    else:
                        date = pd.to_datetime(date, dayfirst=False, yearfirst=False)
                else:
                    print(txt)
                    raise ValueError('Unknown format for date.')
            else:
                obj = yearfirst_regexp.search(txt)
                if obj:
                    date = obj.group(0)
                    date = pd.to_datetime(date, yearfirst=True)
                else:
                    obj = dayfirst_regexp.search(txt)
                    if obj:
                        if len(obj.group('year')) == 2:
                            year_str = txt[-2:]
                            year_int = int(year_str)
                            if year_int <= 22:
                                year_int += 2000
                            else:
                                year_int += 1900
                            year_str = str(year_int)
                            date = txt[:-2] + year_str
                        else:
                            date = obj.group(0)
                        date = pd.to_datetime(date, dayfirst=True)
                    else:
                        print(txt)
                        raise ValueError('Unknown format for date.')
            return date

        date_sep_regexp = re.compile('[ ]*[,;]+[ ]*')

        max_date = datetime.datetime(2000,1,1)
        dc_id_to_inf = {}

        for index_up, row_up in df_up.iterrows():
            #=========================ID
            id1 = row_up['ID1']
            if pd.notnull(id1):
                id1 = extract_id(id1)
                ID = id1
            else:
                id2 = row_up['ID2']
                id2 = extract_id(id2)
                if id2:
                    ID = id2
                else:
                    print(row_up)
                    raise ValueError('Unable to extract ID')
            df_up.loc[index_up, 'ID'] = ID
            print(f'>>>>>>>>>>>{ID=}')
            #=========================DOR
            dor = row_up[DOR]
            #print(dor)
            dor = pd.to_datetime(dor)
            #=========================Reason
            reason = row_up['Reason']
            if pd.notnull(reason):
                if reason.lower() in self.MPD_obj.removal_states_l:
                    pass
                else:
                    raise ValueError('Unknown reason')
            #=========================DOB
            dob = row_up['DOB']
            if pd.notnull(dob):
                dob = convert_str_to_date(dob)
                if max_date < dob:
                    dob = dob - pd.DateOffset(years=100)
                    print('Removing 100 years from dob.')
                df_up.loc[index_up, 'DOB'] = dob
                print(f'{dob=}')
            #=========================Infections
            inf_str = row_up[linf]
            if pd.notnull(inf_str):
                obj = date_sep_regexp.search(inf_str)
                L = []
                if obj:
                    inf_list = inf_str.replace(obj.group(0), ' ').split()
                    for date in inf_list:
                        inf = convert_str_to_date(date)
                        L.append(inf)
                else:
                    inf = convert_str_to_date(inf_str)
                    L.append(inf)
                print(L)
                #We store the list of infections in the dictionary.
                dc_id_to_inf[ID] = L
            #=========================Vaccines
            for date_col, type_col in zip(self.LIS_obj.vaccine_date_cols,
                    self.LIS_obj.vaccine_type_cols):
                d_up = row_up[date_col]
                t_up = row_up[type_col]
                if pd.notnull(d_up):
                    d_up = convert_str_to_date(d_up)
                    df_up.loc[index_up, date_col] = d_up
                if pd.notnull(t_up):
                    t_up = t_up.strip()
                    if t_up in self.LIS_obj.list_of_valid_vaccines:
                        df_up.loc[index_up, type_col] = t_up
                    else:
                        print(f'{t_up=}')
                        raise ValueError('Unknown vaccine type')
        df_up.drop(columns=[linf, 'ID1', 'ID2'], inplace=True)
        columns_with_dates = [DOR, 'DOB'] + self.LIS_obj.vaccine_date_cols
        for column in columns_with_dates:
            df_up[column] = pd.to_datetime(df_up[column])
        self.print_column_and_datatype(df_up)

        #Storing the reformatted update.
        fname  = 'Taras_update_reformatted.xlsx'
        folder = 'Tara_nov_07_2022'
        fname = os.path.join('..','requests',folder, fname)
        df_up.to_excel(fname, index=False)

        #Mergin step.
        self.df = self.merge_with_M_and_return_M(df_up, 'ID', kind='update+')
        #>>>>>>>>>>Infections
        df_inf = pd.DataFrame.from_dict(dc_id_to_inf,
                orient='index').reset_index(level=0)
        df_inf.rename(columns={'index':'ID', 0:1, 1:2, 2:3}, inplace=True)
        df_inf = pd.melt(df_inf, id_vars='ID',
                value_vars=df_inf.columns[1:])
        df_inf.dropna(subset='value', inplace=True)
        df_inf.rename(columns={'variable':'Inf #',
            'value':'DOI'},
            inplace=True)
        print(df_inf)

        #Storing the extracted infections in a separate file.
        fname  = 'extracted_infections_from_Taras_update.xlsx'
        folder = 'Tara_nov_07_2022'
        fname = os.path.join('..','requests',folder, fname)
        df_inf.to_excel(fname, index=False)

        #This has to be executed after the merging process
        #in case we have new participants.
        self.LIS_obj.update_the_dates_and_waves(df_inf)
        self.LIS_obj.order_infections_and_vaccines()




    def tara_nov_07_2022_part_2(self):
        #Loading the udated
        fname  = 'Taras_update_reformatted.xlsx'
        folder = 'Tara_nov_07_2022'
        fname = os.path.join('..','requests',folder, fname)
        df_up = pd.read_excel(fname)
        print('Checking the order of the vaccines.')
        self.LIS_obj.set_chronological_order(df_up,
                self.LIS_obj.vaccine_date_cols[:-1],
                self.LIS_obj.vaccine_type_cols[:-1],
                'Vaccines')

        #Mergin step.
        print('Merging with Taras update')
        self.df = self.merge_with_M_and_return_M(df_up, 'ID', kind='update+')

        #Loading the extracted infections in a separate file.
        fname  = 'extracted_infections_from_Taras_update.xlsx'
        folder = 'Tara_nov_07_2022'
        fname = os.path.join('..','requests',folder, fname)
        df_inf = pd.read_excel(fname)

        #This has to be executed after the merging process
        #in case we have new participants.
        print('Merging with infection update')
        self.LIS_obj.update_the_dates_and_waves(df_inf)
        self.LIS_obj.order_infections_and_vaccines()

    def tara_nov_09_2022(self):
        #Rename Reasons
        store_reformatted_update = True
        #for old_reason, new_reason in zip(self.MPD_obj.removal_states,
                #self.MPD_obj.new_removal_states):
            #self.df['Reason'].replace(old_reason, new_reason, inplace=True)
        fname  = 'update_amica.xlsx'
        folder = 'Tara_nov_09_2022'
        fname = os.path.join('..','requests',folder, fname)
        linf = 'Infections'
        #Read as columns of strings
        df_up = pd.read_excel(fname, dtype=str)
        df_up.replace('n/a', np.nan, inplace=True)
        df_up.replace('N/A', np.nan, inplace=True)
        df_up.replace('refused', np.nan, inplace=True)
        df_up.replace('Refused', np.nan, inplace=True)
        df_up.replace('REFUSED', np.nan, inplace=True)
        df_up.replace('None', np.nan, inplace=True)
        df_up.replace('Unknown', np.nan, inplace=True)
        df_up.replace('Yes - date unknown', np.nan, inplace=True)
        df_up.replace('COVISHEILD', 'COVISHIELD', inplace=True)
        df_up.replace(' ', np.nan, inplace=True)
        df_up.replace('BmodernaO', 'BModernaO', inplace=True)
        print(df_up)
        DOR=self.MPD_obj.DOR

        id_regexp = re.compile('(?P<site>[0-9]{2})[-]?(?P<code>[0-9]{4}[ ]?[0-9]{3})')
        def extract_id(txt):
            obj = id_regexp.search(txt)
            if obj:
                site = obj.group('site')
                code = obj.group('code')
                code = code.replace(' ','')
                ID   = site + '-' + code
                return ID
            else:
                return None

        yearfirst_regexp = re.compile('[0-9]{4}[-][0-9]{2}[-][0-9]{2}')
        dayfirst_regexp = re.compile('[0-9]+[-][a-zA-Z]+[-](?P<year>[0-9]+)')
        txt = ('(?P<month>[0-9]+)' + '[/]' +
        '(?P<day>[0-9]+)' + '[/]' +
        '(?P<year>[0-9]+)')
        monthfirst_regexp = re.compile(txt)
        txt = ('(?P<month>[a-zA-Z]+)' + '[ ]+' +
        '(?P<day>[0-9]{1,2})' + '[,]?' + '[ ]+' +
        '(?P<year>[0-9]{2,})')
        monthfirst_as_text_regexp = re.compile(txt)
        def convert_str_to_date(txt):
            if pd.isnull(txt):
                raise ValueError('Object is NAN')
            obj = monthfirst_as_text_regexp.search(txt)
            if obj:
                #Month(text)/Day/Year
                date = obj.group(0)
                date = pd.to_datetime(date, dayfirst=False, yearfirst=False)
            elif '/' in txt:
                #Month(number)/Day/Year
                obj = monthfirst_regexp.search(txt)
                if obj:
                    date = obj.group(0)
                    month_str = obj.group('month')
                    month_int = int(month_str)
                    if 12 < month_int:
                        date = pd.to_datetime(date, dayfirst=True)
                    else:
                        date = pd.to_datetime(date, dayfirst=False, yearfirst=False)
                else:
                    print(txt)
                    raise ValueError('Unknown format for date.')
            else:
                obj = yearfirst_regexp.search(txt)
                if obj:
                    date = obj.group(0)
                    date = pd.to_datetime(date, yearfirst=True)
                else:
                    obj = dayfirst_regexp.search(txt)
                    if obj:
                        if len(obj.group('year')) == 2:
                            year_str = txt[-2:]
                            year_int = int(year_str)
                            if year_int <= 22:
                                year_int += 2000
                            else:
                                year_int += 1900
                            year_str = str(year_int)
                            date = txt[:-2] + year_str
                        else:
                            date = obj.group(0)
                        date = pd.to_datetime(date, dayfirst=True)
                    else:
                        print(txt)
                        raise ValueError('Unknown format for date.')
            return date

        date_sep_regexp = re.compile('[ ]*[,;]+[ ]*')

        max_date = datetime.datetime(2000,1,1)
        dc_id_to_inf = {}

        check_ID     = False
        check_DOR    = True
        check_reason = True
        for index_up, row_up in df_up.iterrows():
            #=========================ID
            if check_ID:
                id1 = row_up['ID1']
                if pd.notnull(id1):
                    id1 = extract_id(id1)
                    ID = id1
                else:
                    id2 = row_up['ID2']
                    id2 = extract_id(id2)
                    if id2:
                        ID = id2
                    else:
                        print(row_up)
                        raise ValueError('Unable to extract ID')
                df_up.loc[index_up, 'ID'] = ID
            else:
                ID = row_up['ID']
            print(f'>>>>>>>>>>>{ID=}')
            #=========================Reason
            if check_reason:
                if 'Reason' not in df_up.columns:
                    df_up['Reason'] = np.nan
                #Assume that the reason is in a column
                #named Reason+DOR
                RpDOR = 'Reason+DOR'
                rpdor = row_up[RpDOR]
                if pd.notnull(rpdor):
                    found_flag = False
                    for removal_state in self.MPD_obj.removal_states:
                        if removal_state in rpdor:
                            found_flag = True
                            df_up.loc[index_up, 'Reason'] = removal_state
                            print(f'{removal_state=}')
                            break
                    if not found_flag:
                        raise ValueError('Unknown reason')
            else:
                reason = row_up['Reason']
                if pd.notnull(reason):
                    found_flag = False
                    reason = reason.lower()
                    for k, removal_state in enumerate(self.MPD_obj.removal_states_l):
                        if reason in removal_state:
                            found_flag = True
                            reason = self.MPD_obj.removal_states[k]
                            df_up.loc[index_up,'Reason'] = reason
                            break
                    if not found_flag:
                        raise ValueError('Unknown reason')
            #=========================DOR
            if check_DOR:
                if DOR not in df_up.columns:
                    df_up['Reason'] = np.nan
                #Assume that the DOR is in a column
                #named Reason+DOR
                RpDOR = 'Reason+DOR'
                rpdor = row_up[RpDOR]
                if pd.notnull(rpdor):
                    dor = convert_str_to_date(rpdor)
                    df_up.loc[index_up, DOR] = dor
                    print(f'{dor=}')
            else:
                #If DOR is np.nan, then we obtain NaT
                #We assume there should be no problems
                #with the format.
                dor = row_up[DOR]
                dor = pd.to_datetime(dor)
                df_up.loc[index_up, DOR] = dor
            #=========================DOB
            if 'DOB' in df_up.columns:
                dob = row_up['DOB']
                if pd.notnull(dob):
                    dob = convert_str_to_date(dob)
                    if max_date < dob:
                        dob = dob - pd.DateOffset(years=100)
                        print('Removing 100 years from dob.')
                    df_up.loc[index_up, 'DOB'] = dob
                    print(f'{dob=}')
            #=========================Infections
            inf_str = row_up[linf]
            if pd.notnull(inf_str):
                obj = date_sep_regexp.search(inf_str)
                L = []
                if obj:
                    inf_list = inf_str.replace(obj.group(0), ' ').split()
                    for date in inf_list:
                        inf = convert_str_to_date(date)
                        L.append(inf)
                else:
                    inf = convert_str_to_date(inf_str)
                    L.append(inf)
                print(f'Infection list {L=}')
                #We store the list of infections in the dictionary.
                dc_id_to_inf[ID] = L
            #=========================Vaccines
            for date_col, type_col in zip(self.LIS_obj.vaccine_date_cols,
                    self.LIS_obj.vaccine_type_cols):
                d_up = row_up[date_col]
                t_up = row_up[type_col]
                if pd.notnull(d_up):
                    d_up = convert_str_to_date(d_up)
                    df_up.loc[index_up, date_col] = d_up
                if pd.notnull(t_up):
                    t_up = t_up.strip()
                    if t_up in self.LIS_obj.list_of_valid_vaccines:
                        df_up.loc[index_up, type_col] = t_up
                    else:
                        print(f'{t_up=}')
                        raise ValueError('Unknown vaccine type')
        columns_to_drop = []
        for column in df_up.columns:
            if column not in self.df.columns:
                columns_to_drop.append(column)
        df_up.drop(columns=columns_to_drop, inplace=True)
        columns_with_dates = [DOR] + self.LIS_obj.vaccine_date_cols
        if 'DOB' in df_up.columns:
            columns_with_dates.append('DOB')
        for column in columns_with_dates:
            df_up[column] = pd.to_datetime(df_up[column])
        self.print_column_and_datatype(df_up)

        #Date chronology
        print('Checking vaccination chronology.')
        self.LIS_obj.set_chronological_order(df_up,
                self.LIS_obj.vaccine_date_cols[:-1],
                self.LIS_obj.vaccine_type_cols[:-1],
                'Vaccines')

        #Storing the reformatted update.
        if store_reformatted_update:
            fname  = 'Taras_update_reformatted.xlsx'
            folder = 'Tara_nov_09_2022'
            fname = os.path.join('..','requests',folder, fname)
            df_up.to_excel(fname, index=False)
            print(f'Wrote {fname=} to file.')

        #Merging step.
        #Be careful with the kind of update you want to execute.
        #If necessary, you can first run it with "update+"
        #and see if there are significant differences.
        self.df = self.merge_with_M_and_return_M(df_up, 'ID', kind='original+')


        self.LIS_obj.order_infections_and_vaccines()
        #>>>>>>>>>>Infections
        df_inf = pd.DataFrame.from_dict(dc_id_to_inf,
                orient='index').reset_index(level=0)
        dc = {'index':'ID', 0:1, 1:2, 2:3, 3:4, 4:5, 5:6}
        df_inf.rename(columns=dc, inplace=True)
        df_inf = pd.melt(df_inf, id_vars='ID',
                value_vars=df_inf.columns[1:])
        df_inf.dropna(subset='value', inplace=True)
        df_inf.rename(columns={'variable':'Inf #',
            'value':'DOI'},
            inplace=True)
        print(df_inf)

        #Storing the extracted infections in a separate file.
        if store_reformatted_update:
            fname  = 'extracted_infections_from_Taras_update.xlsx'
            folder = 'Tara_nov_09_2022'
            fname = os.path.join('..','requests',folder, fname)
            df_inf.to_excel(fname, index=False)

        #This has to be executed after the merging process
        #in case we have new participants.
        self.LIS_obj.update_the_dates_and_waves(df_inf)
        self.LIS_obj.order_infections_and_vaccines()
        self.MPD_obj.update_active_status_column()


    def lindsay_nov_14_2022(self):
        folder = 'Lindsay_nov_10_2022'
        fname = 'update.xlsx'
        fname = os.path.join(self.requests_path, folder, fname)
        df_up = pd.read_excel(fname, usecols='A:C')
        df_up.replace('?', np.nan, inplace=True)
        df_up['Reason'] = df_up['Reason'].str.replace('WITHDRAW', 'WITHDREW')
        df_up.dropna(axis=0, subset='ID', inplace=True)
        self.check_id_format(df_up, 'ID')
        DOR = self.MPD_obj.DOR
        for index_up, row_up in df_up.iterrows():
            reason = row_up['Reason']
            if pd.notnull(reason):
                reason = reason.lower()
                found_flag = False
                for k,r_state in enumerate(self.MPD_obj.removal_states_l):
                    if r_state in reason:
                        reason = self.MPD_obj.removal_states[k]
                        df_up.loc[index_up, 'Reason'] = reason
                        found_flag = True
                        break
                if not found_flag:
                    print(f'{reason=} is unknown.')
                    df_up.loc[index_up, 'Reason'] = np.nan
        dor = row_up[DOR]
        if pd.notnull(dor):
            dor = pd.to_datetime(dor)
            df_up.loc[index_up, DOR] = dor

        print(df_up)
        self.print_column_and_datatype(df_up)
        self.df = self.merge_with_M_and_return_M(df_up, 'ID', kind='original+')
        self.MPD_obj.update_active_status_column()


    def tara_nov_17_2022(self):
        fname  = 'updates_one_column.xlsx'
        folder = 'Tara_nov_17_2022'
        df_up = self.load_single_column_df_for_update(fname, folder)
        self.extract_and_update_DOR_Reason_Infection(df_up)


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
#Oct 12 2022
#obj.update_LSM()
#obj.update_master_using_SID()
#obj.check_LSM_dates()
#obj.REP_obj.merge_and_plot_Ab_data()
#obj.REP_obj.find_short_jumps()
#Oct 14 2022
#obj.extract_and_update_DOR_Reason_Infection()
#Oct 15 2022
#obj.tara_req_oct_13_2022()
#obj.extract_date_and_method()
#obj.LIS_obj.order_infections_and_vaccines()
#obj.write_the_M_file_to_excel()
#Oct 19 2022
#obj.update_master_using_SID()
#obj.MPD_obj.update_M_from_comments_and_dates()
#Oct 20 2022
#obj.LIS_obj.get_serology_dates_for_infection_dates()
#obj.LIS_obj.compute_slopes_for_serology()
#obj.REP_obj.plot_serology_slopes_from_selection()
#obj.REP_obj.plot_serology_slope_progression()
#obj.REP_obj.plot_serology_slope_vs_days_after_infection()
#obj.REP_obj.plot_serology_slope_vs_bins_after_infection()
#Oct 25/26 2022
#obj.check_zains()
#obj.MPD_obj.compute_age_from_dob()
#obj.write_the_M_file_to_excel()
#Oct 26 2022
#obj.update_master_using_SID()
#obj.write_the_M_file_to_excel()
#Oct 26 2022
#obj.update_LSM()
#obj.LSM_obj.write_to_excel()
#obj.check_LSM_dates()
#obj.LSM_obj.write_to_excel()
#obj.MPD_obj.update_active_status_column()
#obj.LIS_obj.update_PCR_and_infection_status()
#obj.write_the_M_file_to_excel()
#obj.merge_M_with_LSM()
#obj.MPD_obj.missing_DOR()
#Oct 27 2022
#obj.LIS_obj.get_serology_dates_for_infection_dates()
#obj.LIS_obj.compute_slopes_for_serology()
#Oct 31 2022
#obj.update_master_using_SID()
#obj.write_the_M_file_to_excel()
#obj.whole_blood_update()
#obj.write_the_M_file_to_excel()
#obj.LND_obj.clean_LND_file()
#obj.jessica_oct_31_2022()
