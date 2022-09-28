#JRR @ McMaster University
#Update: 18-Sep-2022
import sys
import os
import re
import pandas as pd
import numpy as np
import shutil
import datetime

sys.path.insert(1,'../Master_Participant_Data/')
sys.path.insert(1,'../LTC_Infection_Summary/')
sys.path.insert(1,'../Sample_Inventory_Data/')
sys.path.insert(1,'../LTC_Serology_Master/')

import Master_Participant_Data
import LTC_Infection_Summary
import Sample_Inventory_Data
import LTC_Serology_Master

class Merger:
    def __init__(self):

        #Paths for storage and retrieval
        self.outputs_path = os.path.join('..', 'outputs')
        self.backups_path = os.path.join('..', 'backups')
        self.requests_path = os.path.join('..', 'requests')

        self.df = None
        self.M_pure_name = 'M'
        self.M_fname     = self.M_pure_name + '.xlsx'

        #By default we load the M file
        self.load_the_M_file()

        #By default each class gets a pointer to the M file.
        MPD = 'Master_Participant_Data'
        self.MPD_path = os.path.join('..', MPD)
        self.MPD_obj  = Master_Participant_Data.\
                MasterParticipantData(self.MPD_path, self.df)

        LIS = 'LTC_Infection_Summary'
        self.LIS_path = os.path.join('..', LIS)
        self.LIS_obj  = LTC_Infection_Summary.\
                LTCInfectionSummary(self.LIS_path, self.df)

        SID = 'Sample_Inventory_Data'
        self.SID_path = os.path.join('..', SID)
        self.SID_obj  = Sample_Inventory_Data.\
                SampleInventoryData(self.SID_path, self.df)

        LSM = 'LTC_Serology_Master'
        self.LSM_path = os.path.join('..', LSM)
        #self.LSM_obj  = LTC_Serology_Master.LTCSerologyMaster(self.LSM_path)

        print('Merge class has been loaded.')



    def full_run(self):
        self.MPD_obj.full_run()
        self.LIS_obj.full_run()
        self.SID_obj.full_run()

    def load_components_MPD_LTC_SID(self):
        self.MPD_obj.load_main_frame()
        self.LIS_obj.load_main_frame()
        self.SID_obj.load_main_frame()

    def update_master_using_SID(self):
        #This function updates the merged file M with the
        #Sample Inventory Data file provided by Megan.
        #First, we convert that update into a data frame with the
        #desired format estipulated in the SID class.
        self.load_the_M_file()
        #Now we specify the type of update. We are choosing 'full'
        #in view that Megan sometimes corrects her version. If
        #her version is incorrect, then so is ours. In the
        #special case where we only have to fill empty cells,
        #choose 'complement' for the kind of update.
        up_type = 'full'
        self.df = self.SID_obj.update_master(self.df, up_type)
        print('The M file has been updated')
        self.write_the_M_file_to_excel()

    def backup_the_M_file(self):
        fname = self.M_fname
        original = os.path.join(self.outputs_path, fname)
        today = datetime.datetime.now()
        date  = today.strftime('%d_%m_%Y_time_%H_%M_%S')
        bname = self.M_pure_name + '_backup_' + date
        bname += '.xlsx'
        backup   = os.path.join(self.backups_path, bname)
        shutil.copyfile(original, backup)
        print('A backup for the M file has been generated.')

    def write_the_M_file_to_excel(self):
        self.backup_the_M_file()
        fname = self.M_fname
        fname = os.path.join(self.outputs_path, fname)
        self.df.to_excel(fname, index = False)
        print('The M file has been written to Excel')

    def merge_MPD_LIS_SID_components(self):
        A = pd.merge(self.MPD_obj.df, self.LIS_obj.df, on='ID', how='outer')
        self.df = pd.merge(A, self.SID_obj.df, on='ID', how='outer')

    def load_the_M_file(self):
        #Load the merged file M
        print('Make sure that this is a clone of the encrypted version!')
        fname = self.M_fname
        fname = os.path.join(self.outputs_path, fname)
        self.df = pd.read_excel(fname)
        print('MPD_LIS_SID, aka the M file, has been loaded from Excel')




    def extract_and_update_from_text(self):
        #On a first pass you might not want to modify
        #the M file until you are convinced that the
        #text was correctly parsed.
        self.load_the_M_file()
        edit_now = True
        flag_update_active = False
        flag_update_waves  = False
        infection_dictionary = {'ID':[], 'date':[]}
        status_dictionary = {'ID':[],
                             self.MPD_obj.reason:[],
                             'date':[]}
        fname = 'request_22_09_2022.xlsx'
        fname = os.path.join(self.requests_path, fname)
        df_up = pd.read_excel(fname, header=None)
        #print(df_up)
        id_txt = '(?P<ID>[0-9]+[-][0-9]+)'
        state_txt = '(?P<status>([a-zA-Z]+[ ]+)+)'
        date_txt = '(?P<date>[a-zA-Z]+[ ]+[0-9]{1,2}[ ]+[0-9]+)'
        rx = re.compile(id_txt + '[ ]+' + state_txt + date_txt)
        #rx_date = re.compile('[a-zA-Z]+[ ]+[0-9]{1,2}[ ]+[0-9]+')
        #rx_state = re.compile('[a-zA-Z]+[ ]+[0-9]{1,2}[ ]+[0-9]+')
        #We assume the data frame has only one column.
        for txt in df_up[0]:
            print(txt)
            full = rx.search(txt)
            if full is not None:
                ID     = full.group('ID')
                status = full.group('status')
                status = status.strip()
                dt   = full.group('date')
                print(f'{ID=}')
                print(f'{status=}')
                date = pd.to_datetime(dt)
                print(f'{dt}-->{date}')
                #print(date)
                print('---------')
                if edit_now:
                    selector = self.df['ID'] == ID
                    if status.lower() in self.MPD_obj.removal_states_l:
                        #This individual has been removed
                        flag_update_active = True
                        status_dictionary['ID'].append(ID)
                        status_dictionary[self.MPD_obj.reason].append(status)
                        status_dictionary['date'].append(date)
                    elif status.lower() == 'positive':
                        #This individual had an infection
                        flag_update_waves = True
                        infection_dictionary['ID'].append(ID)
                        infection_dictionary['date'].append(date)

            else:
                raise ValueError('Unable to parse string.')

        if flag_update_active:
            df_up = pd.DataFrame(status_dictionary)
            #print(df_up)
            #Pass a reference to the MPD object to update the df.
            self.MPD_obj.initialize_class_with_df(self.df)
            self.MPD_obj.update_reason_dates_and_status(df_up)
        if flag_update_waves:
            df_up = pd.DataFrame(infection_dictionary)
            #print(df_up)
            #Pass a reference to the LIS object to update the df.
            self.LIS_obj.initialize_class_with_df(self.df)
            self.LIS_obj.update_the_dates_and_waves(df_up)
        #Check the changes in the DF
        #df_up = pd.DataFrame(status_dictionary)
        #DOR      = 'Date Removed from Study'
        #for index, row in df_up.iterrows():
            #date   = row['date']
            #ID     = row['ID']
            #reason = row['Reason']
            #selector = self.df['ID'] == ID
            #print(f'{ID=}')
            #print('Compare:', self.df.loc[selector, 'Reason'].values, ',', reason)
            #print('Compare:', self.df.loc[selector, DOR].values, ',', date)
        self.write_the_M_file_to_excel()


    def check_id_format(self):
        self.load_the_M_file()
        S = set()
        rx = re.compile('(?P<site>[0-9]+)[-](?P<user>[0-9]+)')
        for ID in self.df['ID']:
            obj = rx.match(ID)
            if obj is None:
                print(f'Error found with {ID=}')
                raise ValueError('ID is not compliant.')
            else:
                txt = obj.group('user')
                S.add(len(txt))
        print('All IDs have the format ##-#...#')
        print('User length(s):', S)


    def merge_M_with_LSM(self):
        self.load_the_M_file()
        W = pd.merge(self.LSM_obj.df, self.df, on='ID', how='outer')
        self.write_df_to_excel(W)


    def write_df_to_excel(self, df):
        fpure = 'W.xlsx'
        fname = os.path.join(self.outputs_path, fpure)
        df.to_excel(fname, index = False)
        print(f'The {fpure=} file has been written to Excel.')

    def from_site_and_txt_get_ID(self, site, txt):
        if isinstance(site, int):
            s = str(site)
        else:
            s = site
        if len(s) == 1:
            s = '0' + s
        rx = re.compile('[0-9]+[ ]*[0-9]*')
        obj = rx.search(txt)
        if obj:
            h = obj.group(0).replace(" ","")
            ID = s + '-' + h
            return ID
        else:
            raise ValueError('Unable to produce ID.')

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








obj = Merger()
#obj.full_run()
#obj.load_components_MPD_LIS_SID()
#obj.merge_MPD_LIS_SID_components()
#obj.load_the_M_file()
#obj.satisfy_request()
#obj.compute_all_infection_patterns()
#obj.write_sequence_of_infections_to_file()
#obj.write_infection_edges_to_file()
#obj.update_master_using_SID()
#obj.update_active_status_column()
#obj.check_id_format()
#obj.extract_and_update_from_text()
#obj.merge_M_with_LSM()
#obj.satisfy_rainbow_request()
obj.satisfy_consents_20_09_2022_request()
