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

obj = Comparator()
obj.load_the_rainbow()
