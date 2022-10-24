#JRR @ McMaster University
#Update: 10-Oct-2022
import sys
import os
import re
import pandas as pd
import janitor
import numpy as np
import shutil
import datetime

sys.path.insert(1,'../Master_Participant_Data/')
sys.path.insert(1,'../LTC_Infection_Summary/')
sys.path.insert(1,'../LTC_Serology_Master/')
sys.path.insert(1,'../Sample_Inventory_Data/')
sys.path.insert(1,'../LTC_Neutralization_Data/')
sys.path.insert(1,'../Reporter/')

import Master_Participant_Data
import LTC_Infection_Summary
import Sample_Inventory_Data
import LTC_Serology_Master
import LTC_Neutralization_Data
import Reporter

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

        #Replace passing the data frame for a
        #pointer to the parent object.
        MPD = 'Master_Participant_Data'
        self.MPD_path = os.path.join('..', MPD)
        self.MPD_obj  = Master_Participant_Data.\
                MasterParticipantData(self.MPD_path,
                                      self)

        LIS = 'LTC_Infection_Summary'
        self.LIS_path = os.path.join('..', LIS)
        self.LIS_obj  = LTC_Infection_Summary.\
                LTCInfectionSummary(self.LIS_path,
                                    self)

        LSM = 'LTC_Serology_Master'
        self.LSM_path = os.path.join('..', LSM)
        self.LSM_obj  = LTC_Serology_Master.\
                LTCSerologyMaster(self.LSM_path)

        SID = 'Sample_Inventory_Data'
        self.SID_path = os.path.join('..', SID)
        self.SID_obj  = Sample_Inventory_Data.\
                SampleInventoryData(self.SID_path,
                                    self)

        LND = 'LTC_Neutralization_Data'
        self.LND_path = os.path.join('..', LND)
        self.LND_obj  = LTC_Neutralization_Data.\
                LTCNeutralizationData(self.LND_path)

        REP = 'Reporter'
        self.REP_path = os.path.join('..', REP)
        self.REP_obj  = Reporter.Reporter(self.REP_path,
                                          self)

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
        folder = 'Megan_oct_19_2022'
        fname = 'SID.xlsx'
        fname = os.path.join(self.requests_path, folder, fname)
        #We are only interested in the first column (ID=A) and the 
        #Blood draw columns (W to BF).
        df_up = pd.read_excel(fname,
                skiprows=[0,1,2],
                usecols='A,W:BF',
                sheet_name='All Sites - AutoFill')
        #First, we convert that update into a data frame with the
        #desired format estipulated in the SID class.
        self.SID_obj.format_megans_update(df_up)
        print(df_up)
        #Now we specify the type of update. 
        #The update kind is: update+
        #This means that the update is given higher priority, but
        #it will not erase a cell.
        #To fully replace the entries with the update use
        #update++.
        #In case we only have to fill empty cells,
        #choose 'original+'.
        M = self.merge_with_M_and_return_M(df_up, 'ID', kind='original+')
        print('Merging SID update with M file is complete.')
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


    def extract_and_update_infection_detection_method(self):
        #On a first pass you might not want to modify
        #the M file until you are convinced that the
        #text was correctly parsed.
        #This function updates:
        #LIS
        #by identifying the type of method that was used to detect
        #the infection. Then, it uses the date to check if that
        #infection was already registered, and subsequently 
        #updates the wave data.
        #Examples
        #11-9244 986 • 9/11/2022  (PCR confirmed positive)
        infection_dictionary = {'ID':[], 'date':[], 'method':[]}
        fname = 'part_1.xlsx'
        folder = 'req_Megan_29_Sep_2022'
        fname = os.path.join(self.requests_path, folder, fname)
        df_up = pd.read_excel(fname, sheet_name='method', header=None)
        df_up.dropna(inplace=True)
        print(df_up)
        method_rx = re.compile('(?P<method>[a-zA-Z]+)[ ]confirmed positive')
        id_rx = re.compile('[0-9]{2}[-][0-9]+([ ][0-9]+)?')
        date_rx = re.compile('[0-9]+\D[0-9]+\D[0-9]+')
        for txt in df_up[0]:
            method_obj = method_rx.search(txt)
            if method_obj:
                method = method_obj.group('method')
                print(f'{method=}')
                match = method_obj.group(0)
                txt2 = txt.replace(match,'')
                id_obj = id_rx.search(txt2)
                if id_obj:
                    ID = id_obj.group(0)
                    txt3 = txt2.replace(ID, '')
                    ID = ID.replace(' ','')
                    print(f'{ID=}')
                    date_obj = date_rx.search(txt3)
                    if date_obj:
                        date = date_obj.group(0)
                        print(f'{date=}')
                    else:
                        raise ValueError('Unable to parse string.')
                else:
                    raise ValueError('Unable to parse string.')
            else:
                raise ValueError('Unable to parse string.')
            print('--------------')
            infection_dictionary['ID'].append(ID)
            infection_dictionary['date'].append(date)
            infection_dictionary['method'].append(method)
        #Create DF
        df_up = pd.DataFrame(infection_dictionary)
        df_up['date'] = pd.to_datetime(df_up['date'])
        print(df_up)
        self.LIS_obj.update_the_dates_and_waves(df_up)
        #self.write_the_M_file_to_excel()




    def extract_and_update_DOR_Reason_Infection(self):
        #On a first pass you might not want to modify
        #the M file until you are convinced that the
        #text was correctly parsed.
        #This function updates:
        #MPD
        #LIS
        #This function is able to identify date of removal, reason,
        #and infection date.
        #Examples
        #50-1910008 Deceased Sep 15 2022
        #14-5077158  Positive Oct 2 2022
        flag_update_active = False
        flag_update_waves  = False
        infection_dictionary = {'ID':[], 'date':[]}
        reason_dictionary = {'ID':[],
                             self.MPD_obj.reason:[],
                             'date':[]}
        fname = 'data.xlsx'
        folder = 'Tara_oct_14_2022'
        fname = os.path.join(self.requests_path, folder, fname)
        df_up = pd.read_excel(fname, header=None)
        df_up.dropna(axis=0, inplace=True)
        id_rx     = re.compile('[0-9]+[-][0-9]+')
        reason_rx = re.compile('[a-zA-Z]+([ ][a-zA-Z]+)*')
        date_rx   = re.compile('[a-zA-Z]+[ ]+[0-9]{1,2}[ ]+[0-9]+')
        #We assume the data frame has only one column.
        for txt in df_up[0]:
            print(txt)
            date_obj = date_rx.search(txt)
            if date_obj:
                date   = date_obj.group(0)
                print(f'{date=}')
                txt_m_date = txt.replace(date, '')
                id_obj = id_rx.search(txt_m_date)
                if id_obj:
                    ID = id_obj.group(0)
                    print(f'{ID=}')
                    txt_m_date_m_id = txt_m_date.replace(ID, '')
                    reason_obj = reason_rx.search(txt_m_date_m_id)
                    if reason_obj:
                        reason = reason_obj.group(0)
                        print(f'{reason=}')
                    else:
                        raise ValueError('Unable to parse string.')
                else:
                    raise ValueError('Unable to parse string.')
            else:
                raise ValueError('Unable to parse string.')

            #print(f'{ID=}')
            #print(f'{reason=}')
            #print(f'{date=}')
            print('---------Extraction is complete.')
            selector = self.df['ID'] == ID
            if reason.lower() in self.MPD_obj.removal_states_l:
                #This individual has been removed
                flag_update_active = True
                reason_dictionary['ID'].append(ID)
                reason_dictionary[self.MPD_obj.reason].append(reason)
                reason_dictionary['date'].append(date)
            elif reason.lower() == 'positive':
                #This individual had an infection
                flag_update_waves = True
                infection_dictionary['ID'].append(ID)
                infection_dictionary['date'].append(date)

        if flag_update_active:
            df_up = pd.DataFrame(reason_dictionary)
            df_up['date'] = pd.to_datetime(df_up['date'])
            print(df_up)
            self.MPD_obj.update_reason_dates_and_status(df_up)
        if flag_update_waves:
            df_up = pd.DataFrame(infection_dictionary)
            df_up['date'] = pd.to_datetime(df_up['date'])
            print(df_up)
            self.LIS_obj.update_the_dates_and_waves(df_up)
        print('Not currently writing')
        #self.write_the_M_file_to_excel()


    def check_id_format(self, df, col):
        #Modified to be applicable to any df and given column.
        user_length_set = set()
        site_length_set = set()
        rexp = re.compile('(?P<site>[0-9]+)[-](?P<user>[0-9]+)')
        def is_id(txt):
            if isinstance(txt, str):
                obj = rexp.match(txt)
                if obj:
                    user_length_set.add(len(obj.group('user')))
                    site_length_set.add(len(obj.group('site')))
                    return True
                else:
                    return False
            else:
                return False

        selector = df[col].apply(is_id)
        if ~selector.all():
            print('Not all IDs are compliant.')
        else:
            print('All IDs have the format ##-#...#')
            print(f'{user_length_set=}, {site_length_set=}')


    def merge_M_with_LSM(self):
        self.load_the_M_file()
        W = pd.merge(self.LSM_obj.df, self.df, on='ID', how='outer')
        self.write_df_to_excel(W)


    def write_df_to_excel(self, df):
        fpure = 'infection_dates_delta.xlsx'
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



    def print_column_and_datatype(self, df):
        #Sep 29, 2022
        #This function is commonly used and now
        #has a separate definition to avoid code
        #repetition.
        for col, dtype in zip(df.columns, df.dtypes):
            print(f'{col:35}: {dtype}')

    def merge_with_M_and_return_M(self, df_up, merge_column, kind='update+'):
        #Oct 12, 2022
        #This function is commonly used and now
        #has a separate definition to avoid code
        #repetition.
        #Merge on: merge_column
        #Outer-type merge
        #We check that all columns are common.
        #If the kind is 'correct', then we give priority to the update.
        #If the kind is 'complement', we prioritize the M file 
        #and only fill empty cells.

        M = pd.merge(self.df, df_up, on=merge_column, how='outer')
        for column in df_up.columns:
            if column not in self.df.columns:
                print(column)
                raise ValueError('All columns in the update should be common')
            if column == merge_column:
                continue
            left  = column + '_x'
            right = column + '_y'
            #Check if we have new data
            is_new_data = M[left].isnull() & M[right].notnull()
            if is_new_data.any():
                print(f'{column=} has new information.')
            else:
                print(f'{column=} was already complete.')
            if kind == 'update++':
                #Only trust the update
                M[column] = M[right]
            elif kind == 'update+':
                #The update has a higher priority.
                #Keep the update if not empty.
                #Otherwise, use the original.
                M[column] = M[right].where(M[right].notnull(), M[left])
            elif kind == 'original+':
                #The original has a higher priority.
                #Keep the original if not empty.
                #Otherwise, use the update.
                M[column] = M[left].where(M[left].notnull(), M[right])
            else:
                raise ValueError('Unexpected kind for the update.')
            M.drop(columns=[left, right], inplace=True)
        return M[self.df.columns]


    def update_LSM(self):
        #This function was updated on 12-Oct-2022
        #fname = 'june_ltc.xlsx'
        fname = 'nc_sept.xlsx'
        folder = 'jbreznik_oct_12_2022'
        fname = os.path.join('..','requests',folder, fname)
        book = pd.read_excel(fname, sheet_name=None, header=None)
        print(f'LSM is looking into the {folder=}')
        print(f'LSM is opening the {fname=}')
        for k, (sheet, df_up) in enumerate(book.items()):
            if k == 2:
                pass
            if k == 0 or k == 1:
                continue
            print('>>>>>>>>',k)
            print(f'Updating using {sheet=}.')
            print(df_up)
            self.LSM_obj.merge_serology_update(df_up)
        #Uncomment the following line if you want to verify
        #that the LSM dates are consistent with the SID file.
        #self.SID_obj.check_LSM_dates_using_SID()

        self.LSM_obj.write_to_excel()

        #want_to_print = input('Are you sure you want to overwrite the LSM? ')
        #if want_to_print == 'y':
            #print('Writing to LSM >>>')
        #else:
            #print('Writing aborted.')

    def check_LSM_dates(self):
        #This function was updated on 12-Oct-2022
        #This function makes sure that all the dates
        #in the LSM file are consistent with the 
        #dates in the SID file.
        self.SID_obj.check_LSM_dates_using_SID()




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
#obj.missing_dates()
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
obj.REP_obj.plot_serology_slope_vs_days_after_infection()
