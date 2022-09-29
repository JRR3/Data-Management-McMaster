#JRR @ McMaster University
#Update: 28-Sep-2022
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
        #Note that this is not the case for the LTC_Serology_Master
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
        self.LSM_obj  = LTC_Serology_Master.\
                LTCSerologyMaster(self.LSM_path)

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


    def update_LSM(self):
        fname = 'ab_data.xlsx'
        folder = 'Antibody Data Update - September 25 2022'
        fname = os.path.join('..','requests',folder, fname)
        book = pd.read_excel(fname, sheet_name=None, header=None)
        for sheet, df_up in book.items():
            print(f'Updating using {sheet=}.')
            self.LSM_obj.merge_serology_update(df_up)
        self.LSM_obj.write_to_excel()

    def request_jbreznik_26_09_2022(self):
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
#obj.satisfy_consents_20_09_2022_request()
#obj.update_LSM()
obj.request_jbreznik_26_09_2022()
