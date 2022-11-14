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
#sys.path.insert(1,'../LTC_Neutralization_Data/')
sys.path.insert(1,'../Reporter/')

import Master_Participant_Data
import LTC_Infection_Summary
import Sample_Inventory_Data
import LTC_Serology_Master
#import LTC_Neutralization_Data
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

        self.merge_source = 'Full ID'
        self.merge_column = 'ID'

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
                LTCSerologyMaster(self.LSM_path,
                                  self)

        SID = 'Sample_Inventory_Data'
        self.SID_path = os.path.join('..', SID)
        self.SID_obj  = Sample_Inventory_Data.\
                SampleInventoryData(self.SID_path,
                                    self)

        #The LSM class now controls this section.
        #LND = 'LTC_Neutralization_Data'
        #self.LND_path = os.path.join('..', LND)
        #self.LND_obj  = LTC_Neutralization_Data.\
                #LTCNeutralizationData(self.LND_path,
                                      #self)

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
        folder = 'Megan_nov_11_2022'
        fname = 'sid_update.xlsx'
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
        #Rewrite the self.df object with the M data frame.
        self.df = self.merge_with_M_and_return_M(df_up, 'ID', kind='original+')
        print('Merging SID update with M file is complete.')
        #Note==>
        #The writing to Excel should be executed externally.

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
        #Load the Master file M
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


    def load_single_column_df_for_update(self, fname, folder, sheet=0):
        #Nov 2 2022
        fname = os.path.join(self.requests_path, folder, fname)
        df_up = pd.read_excel(fname, header=None, sheet_name=sheet)
        #df_up.dropna(axis=0, inplace=True)
        return df_up

    def generate_infection_and_reason_dict(self, df_up):
        flag_update_active = False
        flag_update_waves  = False
        infection_dictionary = {'ID':[], 'date':[]}
        reason_dictionary = {'ID':[],
                             self.MPD_obj.reason:[],
                             'date':[]}
        #fname = 'up.xlsx'
        #folder = 'Tara_oct_31_2022'
        #fname = os.path.join(self.requests_path, folder, fname)
        #df_up = pd.read_excel(fname, header=None)
        #df_up.dropna(axis=0, inplace=True)

        id_rx     = re.compile('[0-9]+[-][0-9]+')
        #The following regexp is no longer in use.
        #reason_rx = re.compile('[a-zA-Z]+([ ][a-zA-Z]+)*')
        #Moved Out --> Moved
        #Now we use the following.
        reason_rx = re.compile('[a-zA-Z]+')
        date_rx   = re.compile('[a-zA-Z]+[ ]+[0-9]{1,2}(?P<year>[ ]+[0-9]+)?')
        #We assume the data frame has only one column.
        for txt in df_up[0]:
            print(txt)
            date_obj = date_rx.search(txt)
            if date_obj:
                date   = date_obj.group(0)
                txt_m_date = txt.replace(date, '')
                if date_obj.group('year'):
                    pass
                else:
                    #If no year is given, we infer that 2022 is meant.
                    date += ' 2022'
                id_obj = id_rx.search(txt_m_date)
                if id_obj:
                    ID = id_obj.group(0)
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

            print(f'{ID=}')
            print(f'{reason=}')
            print(f'{date=}')
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

        return (flag_update_active,
                flag_update_waves,
                infection_dictionary,
                reason_dictionary)


    def extract_and_update_DOR_Reason_Infection(self, df_up):
        #Modified on Nov 1, 2022
        #Moved Out --> Moved
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
        #14-5077158  Positive Oct 2 (No year)


        (flag_update_active,
                flag_update_waves,
                infection_dictionary,
                reason_dictionary) =\
                        self.generate_infection_and_reason_dict(df_up)

        if flag_update_active:
            df_up = pd.DataFrame(reason_dictionary)
            df_up['date'] = pd.to_datetime(df_up['date'])
            print(df_up)
            self.MPD_obj.update_reason_dates_and_status(df_up)
        if flag_update_waves:
            df_up = pd.DataFrame(infection_dictionary)
            df_up['DOI'] = pd.to_datetime(df_up['date'])
            print(df_up)
            self.LIS_obj.update_the_dates_and_waves(df_up)
        print('Please write to Excel externally.')


    def check_id_format(self, df, col):
        #Modified to be applicable to any df and given column.
        #user_length_set = set()
        #site_length_set = set()
        rexp = re.compile('(?P<site>[0-9]{2})[-](?P<user>[0-9]{7})')
        def is_id(txt):
            if isinstance(txt, str):
                obj = rexp.match(txt)
                if obj:
                    #user_length_set.add(len(obj.group('user')))
                    #site_length_set.add(len(obj.group('site')))
                    #return True
                    pass
                else:
                    print(txt)
                    raise ValueError(f'Unexpected format for {col=}')
            else:
                raise ValueError(f'Unexpected type for {txt=}')

        #selector = df[col].apply(is_id)
        #if ~selector.all():
            #print('Not all IDs are compliant.')
        #else:
            #print('All IDs have the format ##-#...#')
            #print(f'{user_length_set=}, {site_length_set=}')


    def merge_M_with_LSM(self):
        #self.load_the_M_file()
        W = pd.merge(self.LSM_obj.df, self.df, on='ID', how='outer')
        self.write_df_to_excel(W)


    def write_df_to_excel(self, df, label='W'):
        #A data frame df has to be passed as an argument.
        #fpure = 'infection_dates_delta.xlsx'
        fpure = label + '.xlsx'
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
        #Oct 31, 2022
        #This function is commonly used and now
        #has a separate definition to avoid code
        #repetition.
        #Merge on: merge_column
        #Outer-type merge
        #We check that all columns are common.
        #If the kind is 'correct', then we give priority to the update.
        #If the kind is 'complement', we prioritize the M file 
        #and only fill empty cells.
        return self.merge_X_with_Y_and_return_Z(self.df,
                                                df_up,
                                                merge_column,
                                                kind=kind)


    def update_LSM(self):
        #This function was updated on 10-Nov-2022
        fname = 'update.xlsx'
        folder = 'Jessica_nov_14_2022'
        fname = os.path.join('..','requests',folder, fname)
        book = pd.read_excel(fname, sheet_name=None, header=None)
        print(f'LSM is looking into the {folder=}')
        print(f'LSM is opening the {fname=}')
        for k, (sheet, df_up) in enumerate(book.items()):
            print('>>>>>>>>',k)
            print(f'Updating using {sheet=}.')
            #print(df_up)
            self.LSM_obj.merge_serology_update(df_up)

        #Uncomment the following line if you want to verify
        #that the LSM dates are consistent with the SID file.
        #self.SID_obj.check_df_dates_using_SID(self.LSM_obj.df)

        #The writing process should be executed 
        #externally for safety reasons.
        print('Do not forget to write the write the file to Excel.')

    def check_LSM_dates(self):
        #This function was updated on 12-Oct-2022
        #This function makes sure that all the dates
        #in the LSM file are consistent with the 
        #dates in the SID file.
        self.SID_obj.check_df_dates_using_SID(self.LSM_obj.df)

    def merge_X_with_Y_and_return_Z(self, X, Y, merge_column, kind='update+'):
        #Oct 31, 2022
        #This function is commonly used and now
        #has a separate definition to avoid code
        #repetition.
        #Merge on: merge_column
        #Outer-type merge
        #We check that all columns are common.
        #If the kind is 'update+', then we give priority to the update Y.
        #If the kind is 'original+', we prioritize the X file 
        #and only fill empty cells.
        Z = pd.merge(X, Y, on=merge_column, how='outer')
        for column in Y.columns:
            if column not in X.columns:
                print(column)
                raise ValueError('All columns in the update should be common.')
            if column == merge_column:
                continue
            left  = column + '_x'
            right = column + '_y'
            #Check if we have new data
            is_new_data = Z[left].isnull() & Z[right].notnull()
            if is_new_data.any():
                print(f'{column=} has new information.')
            else:
                print(f'{column=} was already complete.')
            if kind == 'update++':
                #Only trust the update
                Z[column] = Z[right]
            elif kind == 'update+':
                #The update has a higher priority.
                #Keep the update if not empty.
                #Otherwise, use the original.
                Z[column] = Z[right].where(Z[right].notnull(), Z[left])
            elif kind == 'original+':
                #The original has a higher priority.
                #Keep the original if not empty.
                #Otherwise, use the update.
                Z[column] = Z[left].where(Z[left].notnull(), Z[right])
            else:
                raise ValueError('Unexpected kind for the update.')
            Z.drop(columns=[left, right], inplace=True)
        return Z[X.columns]

    def create_df_with_ID_from_full_ID(self, Y):
        #This function was updated on Oct 31, 2022
        #For safety reasons we create a copy of the 
        #original data frame.
        #This function is used inside the LND and LSM
        #classes.
        X = Y.copy()
        flag_needs_order = False
        if self.merge_column not in X.columns:
            print('Creating the ID column.')
            flag_needs_order = True
            column_order = [self.merge_column] + X.columns.to_list()
            X[self.merge_column] = ''
        else:
            print('The ID column already exists.')
        id_rx = re.compile('(?P<ID>[0-9]{2}[-][0-9]{7})[-][A-Z]{1}')

        def get_id_from_full_id(txt):
            obj = id_rx.match(txt)
            if obj is None:
                print(txt)
                raise ValueError('ID is not compliant.')
            return obj.group('ID')

        X[self.merge_column] =\
                X[self.merge_source].apply(get_id_from_full_id)
        print('ID column has been updated inside the X file.')

        if flag_needs_order:
            print('Reordering the columns.')
            X = X[column_order]

        if X[self.merge_source].value_counts().gt(1).any():
            raise ValueError('No repetitions should be present.')
        print('The merge operation is complete. Returning merged DF.')
        return X


    def tara_nov_11_2022(self):
        #Use this function to update the Master file
        #when using Retirement Home data.
        store_reformatted_update = True
        #Rename Reasons
        #for old_reason, new_reason in zip(self.MPD_obj.removal_states,
                #self.MPD_obj.new_removal_states):
            #self.df['Reason'].replace(old_reason, new_reason, inplace=True)
        fname  = 'site_01_update.xlsx'
        folder = 'Tara_nov_11_2022'
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

        dayfirst_with_slash_regexp = re.compile('[0-9]+[/][0-9]+[/](?P<year>[0-9]+)')

        txt = ('(?P<month>[0-9]+)' + '[/]' +
        '(?P<day>[0-9]+)' + '[/]' +
        '(?P<year>[0-9]+)')
        monthfirst_regexp = re.compile(txt)

        txt = ('(?P<month>[a-zA-Z]+)' + '[ ]+' +
        '(?P<day>[0-9]{1,2})' + '[a-z]*' +
        '[,/]?' + '[ ]*' +
        '(?P<year>[0-9]{2,})')
        monthfirst_as_text_regexp = re.compile(txt)

        def short_year_to_long_year(obj):
            date = obj.group(0)
            if len(obj.group('year')) == 2:
                year_str = date[-2:]
                year_int = int(year_str)
                if year_int <= 22:
                    year_int += 2000
                else:
                    year_int += 1900
                year_str = str(year_int)
                date = date[:-2] + year_str
            return date

        def extract_method(txt):
            if 'PCR' in txt:
                return 'PCR'
            elif 'RAT' in txt:
                return 'RAT'
            elif 'DBS' in txt:
                return 'DBS'
            else:
                return None

        def convert_str_to_date(txt, use_day_first_for_slash=False):
            if pd.isnull(txt):
                raise ValueError('Object is NAN')
            obj = monthfirst_as_text_regexp.search(txt)
            if obj:
                #Month(text)/Day/Year
                #Check if we have a short year.
                date = short_year_to_long_year(obj)
                date = pd.to_datetime(date, dayfirst=False, yearfirst=False)
            elif '/' in txt:
                if use_day_first_for_slash:
                    obj = dayfirst_with_slash_regexp.search(txt)
                    if obj:
                        date = obj.group(0)
                        date = pd.to_datetime(date, dayfirst=True)
                    else:
                        raise ValueError('Unexpected date for slash with day first.')
                else:
                    #Month(number)/Day/Year
                    obj = monthfirst_regexp.search(txt)
                    if obj:
                        #Check if we have a short year.
                        date = short_year_to_long_year(obj)
                        month_str = obj.group('month')
                        month_int = int(month_str)
                        if 12 < month_int:
                            print('Unexpected format: Month/Day/Year but Month > 12')
                            date = pd.to_datetime(date, dayfirst=True)
                        else:
                            date = pd.to_datetime(date, dayfirst=False, yearfirst=False)
                    else:
                        print(txt)
                        raise ValueError('Unknown format for date.')
            else:
                #In this case we do not expect a short year.
                obj = yearfirst_regexp.search(txt)
                if obj:
                    date = obj.group(0)
                    date = pd.to_datetime(date, yearfirst=True)
                else:
                    obj = dayfirst_regexp.search(txt)
                    if obj:
                        #Check if we have a short year.
                        date = short_year_to_long_year(obj)
                        date = pd.to_datetime(date, dayfirst=True)
                    else:
                        print(txt)
                        raise ValueError('Unknown format for date.')
            return date

        date_sep_regexp = re.compile('[ ]*[,;]+[ ]*')

        max_date_for_DOB = datetime.datetime(2000,1,1)
        dc_id_to_inf = {}
        dc_id_to_method = {}

        ID_in_ID1_or_ID2          = False
        DOR_is_merged_with_Reason = False
        find_vaccines             = False
        find_infections           = True

        for index_up, row_up in df_up.iterrows():
            #=========================ID
            if ID_in_ID1_or_ID2:
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
            if DOR_is_merged_with_Reason:
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
                if 'Reason' in df_up.columns:
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
            if DOR_is_merged_with_Reason:
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
                if DOR in df_up.columns:
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
                    dob = convert_str_to_date(dob, use_day_first_for_slash=True)
                    if max_date_for_DOB < dob:
                        dob = dob - pd.DateOffset(years=100)
                        print('Removing 100 years from dob.')
                    df_up.loc[index_up, 'DOB'] = dob
                    print(f'{dob=}')
            #=========================Infections
            if find_infections:
                inf_str = row_up[linf]
                if pd.notnull(inf_str):
                    obj = date_sep_regexp.search(inf_str)
                    L = []
                    M = []
                    if obj:
                        #inf_list = inf_str.replace(obj.group(0), ' ').split()
                        inf_list = inf_str.split(',')
                        for inf in inf_list:
                            method = extract_method(inf)
                            inf = convert_str_to_date(inf)
                            L.append(inf)
                            if method:
                                M.append(method)
                    else:
                        method = extract_method(inf_str)
                        inf = convert_str_to_date(inf_str)
                        L.append(inf)
                        if method:
                            M.append(method)
                    print(f'Infection list {L=}')
                    #We store the list of infections in the dictionary.
                    dc_id_to_inf[ID] = L
                    if 0 < len(M):
                        dc_id_to_method[ID] = M
            #=========================Vaccines
            if find_vaccines:
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
        #We only keep columns that appear in the Master file.
        for column in df_up.columns:
            if column not in self.df.columns:
                columns_to_drop.append(column)
        df_up.drop(columns=columns_to_drop, inplace=True)
        columns_with_dates = []
        if 'DOB' in df_up.columns:
            columns_with_dates.append('DOB')
        if 'DOR' in df_up.columns:
            columns_with_dates.append('DOR')
        if find_vaccines:
            columns_with_dates.extend(self.LIS_obj.vaccine_date_cols)
        for column in columns_with_dates:
            df_up[column] = pd.to_datetime(df_up[column])
        self.print_column_and_datatype(df_up)
        print(df_up)

        #Date chronology
        if find_vaccines:
            print('Checking vaccination chronology of the update.')
            self.LIS_obj.set_chronological_order(df_up,
                    self.LIS_obj.vaccine_date_cols[:-1],
                    self.LIS_obj.vaccine_type_cols[:-1],
                    'Vaccines')


        #Storing the reformatted update.
        if store_reformatted_update:
            fname  = 'Taras_update_reformatted.xlsx'
            fname = os.path.join('..','requests',folder, fname)
            df_up.to_excel(fname, index=False)
            print(f'Wrote {fname=} to file.')

        #Merging step.
        #Be careful with the kind of update you want to execute.
        #If necessary, you can first run it with "update+"
        #and see if there are significant differences.
        self.df = self.merge_with_M_and_return_M(df_up, 'ID', kind='original+')


        if find_vaccines:
            print('Checking vaccination chronology of the merged file.')
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
        #>>>>>>>>>>Method
        if 0 < len(dc_id_to_method):
            df_method = pd.DataFrame.from_dict(dc_id_to_method,
                    orient='index').reset_index(level=0)
            dc = {'index':'ID', 0:1, 1:2, 2:3, 3:4, 4:5, 5:6}
            df_method.rename(columns=dc, inplace=True)
            df_method = pd.melt(df_method, id_vars='ID',
                    value_vars=df_method.columns[1:])
            df_method.dropna(subset='value', inplace=True)
            df_method.rename(columns={'variable':'Inf #',
            'value':'Method'},
            inplace=True)
            print(df_method)
        return

        #Storing the extracted infections in a separate file.
        if store_reformatted_update:
            fname  = 'extracted_infections_from_Taras_update.xlsx'
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




obj = Merger()
#Oct 31 2022
#Generate the slope plots respecting the thresholds.
#obj.REP_obj.plot_serology_one_Ig_from_df(Ig='G', max_n_inf=5)
#obj.REP_obj.plot_serology_one_Ig_from_df(Ig='A', max_n_inf=5)
#obj.REP_obj.plot_serology_one_Ig_from_df(Ig='G', max_n_inf=1)
#obj.REP_obj.plot_serology_one_Ig_from_df(Ig='A', max_n_inf=1)
#Oct 31 2022 (Recompute Serology Update)
#obj.LND_obj.clean_LND_file()
#obj.jessica_oct_31_2022()
#obj.LSM_obj.write_to_excel()
#Oct 31 2022 (Tara's update)
#obj.tara_oct_31_2022()
#obj.write_the_M_file_to_excel()
#obj.LIS_obj.compute_waves_of_infection()
#obj.LIS_obj.assume_PCR_if_empty()
#obj.LIS_obj.update_PCR_and_infection_status()
#obj.write_the_M_file_to_excel()
#Nov 02 2022 - Nov 03 2022
#obj.LIS_obj.update_ahmad_file()
#obj.LIS_obj.write_ahmad_df_to_excel()
#obj.compare_ahmad_infection_file_w_M()
#Nov 03 2022
#obj.LIS_obj.get_serology_dates_for_infection_dates()
#obj.LIS_obj.compute_slopes_for_serology()
#obj.LIS_obj.plot_dawns_infection_count()
#Nov 04 2022
#obj.LIS_obj.update_ahmad_file()
#obj.LIS_obj.write_ahmad_df_to_excel()
#Nov 07 2022
#obj.tara_nov_07_2022()
#obj.write_the_M_file_to_excel()
#obj.tara_nov_07_2022_part_2()
#obj.write_the_M_file_to_excel()
#Nov 09 2022
#obj.tara_nov_09_2022()
#obj.write_the_M_file_to_excel()
#Nov 10 2022
#obj.tara_nov_10_2022()
#obj.write_the_M_file_to_excel()
#obj.update_LSM()
#obj.LSM_obj.write_LSM_to_excel()
#obj.update_master_using_SID()
#obj.write_the_M_file_to_excel()
#obj.LSM_obj.update_LND_data()
#obj.LSM_obj.write_LSM_to_excel()
#Nov 11 2022
#obj.update_master_using_SID()
#obj.write_the_M_file_to_excel()
#obj.merge_M_with_LSM()
#obj.LIS_obj.get_serology_dates_for_infection_dates()
#obj.LIS_obj.compute_slopes_for_serology()
#obj.LIS_obj.produce_melted_files()
#obj.LSM_obj.what_is_missing()
#obj.LIS_obj.produce_melted_files()
#Nov 14 2022
#obj.lindsay_nov_14_2022()
#obj.write_the_M_file_to_excel()
#obj.update_LSM()
#obj.LSM_obj.write_LSM_to_excel()
obj.tara_nov_11_2022()
