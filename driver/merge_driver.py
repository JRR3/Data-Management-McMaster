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

        LND = 'LTC_Neutralization_Data'
        self.LND_path = os.path.join('..', LND)
        self.LND_obj  = LTC_Neutralization_Data.\
                LTCNeutralizationData(self.LND_path,
                                      self)

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
        folder = 'Megan_oct_31_2022'
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


    def load_single_column_df_for_update(self, sheet=0):
        #We assume
        fname = 'up.xlsx'
        folder = 'Ahmad_oct_31_2022'
        fname = os.path.join(self.requests_path, folder, fname)
        df_up = pd.read_excel(fname, header=None, sheet_name=sheet)
        df_up.dropna(axis=0, inplace=True)
        return df_up

    def generate_infection_and_reason_dictionaries(self, df_up):
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

        df_up = self.load_single_column_df_for_update()

        (flag_update_active,
                flag_update_waves,
                infection_dictionary,
                reason_dictionary) =\
                        self.generate_infection_and_reason_dictionaries(df_up)

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
        #This function was updated on 12-Oct-2022
        #fname = 'june_ltc.xlsx'
        fname = 'nc_ltc.xlsx'
        folder = 'Jessica_oct_26_2022'
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
        #externally for safery reasons.

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
        id_rx = re.compile('(?P<ID>[0-9]+[-][0-9]+)[-][A-Z]+')

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

    def check_Ahmad_inf_file(self):
        #What is different between Ahmad's file
        #and the master file.
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
        #self.print_column_and_datatype(df_up)
        self.positive_date_cols  = []
        self.positive_type_cols  = []
        for index, row in df_up.iterrows():
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
#Nov 02 2022
#obj.check_Ahmad_inf_file()
obj.LIS_obj.update_ahmad_file()
