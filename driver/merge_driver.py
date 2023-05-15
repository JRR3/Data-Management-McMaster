#JRR @ McMaster University
#Update: 04-May-2023
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
        self.AID = 'Analytics ID'

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
        #Dec 23 2022
        #This version also includes the generation
        #of the delta report.
        self.backup_the_M_file()
        fname = self.M_fname
        fname = os.path.join(self.outputs_path, fname)

        with pd.ExcelWriter(fname) as writer:
            self.df.to_excel(writer,
                    sheet_name = 'data', index = False)
            if self.MPD_obj.delta_report is None:
                pass
            else:
                print('Writing the Delta report to Excel')
                self.MPD_obj.delta_report.to_excel(writer,
                        sheet_name = 'report', index = False)

        print('The M file has been written to Excel')


    def load_the_M_file(self):
        fname = 'column_types.xlsx'
        fname = os.path.join(self.outputs_path, fname)
        df_t = pd.read_excel(fname)
        date_cols = []
        name_to_type = {}
        #Use the corresponding data types for each column.
        for index, row in df_t.iterrows():
            name = row['Name']
            col_type = row['Type']
            if col_type == 'date':
                date_cols.append(name)
            else:
                name_to_type[name] = col_type

        #Load the Master file M
        print('Make sure that this is a clone of the encrypted version!')
        fname = self.M_fname
        fname = os.path.join(self.outputs_path, fname)
        self.df = pd.read_excel(fname, dtype = name_to_type, parse_dates=date_cols)
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
        #date_rx = re.compile('[0-9]+\D[0-9]+\D[0-9]+')
        date_rx = re.compile('[0-9]+[-/][0-9]+[-/][0-9]+')
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





    def check_id_format(self, df, col):
        #March 22 2023
        #Modified to be applicable to any df and given column.
        print('>>>Checking the ID format')
        rexp = re.compile('(?P<site>[0-9]{2})[-](?P<user>[0-9]{7})')
        def is_id(txt):
            if isinstance(txt, str):
                obj = rexp.match(txt)
                if obj:
                    pass
                else:
                    print(txt)
                    raise ValueError(f'Unexpected format for {col=}')
            elif txt is np.nan:
                pass
            else:
                raise ValueError(f'Unexpected type for {txt=}')

        selector = df[col].apply(is_id)
        print('All IDs have the expected format')


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

    def merge_with_M_and_return_M(self, df_up, merge_column, kind='original+'):
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


    def update_LSM(self, file_name=None):
        #This function was updated on Dec 21 2022
        if file_name:
            #Not none
            fname = file_name
        else:
            #None
            fname = 'updates.xlsx'
        folder = 'Jessica_may_02_2023'
        fname = os.path.join('..','requests',folder, fname)
        book = pd.read_excel(fname, sheet_name=None)
        print(f'LSM is looking into the {folder=}')
        print(f'LSM is opening the {fname=}')

        #Before going through the updates, we store the current
        #data density state of the LSM file.
        status_pre = self.MPD_obj.compute_data_density(self.LSM_obj.df)

        #Iterate over the updates.
        for k, (sheet, df_up) in enumerate(book.items()):
            print('>>>>>>>>',k)
            print(f'Updating using {sheet=}.')
            self.LSM_obj.direct_serology_update_with_headers(df_up)
            #print(df_up)
            #The following function is now obsolete.
            #self.LSM_obj.merge_serology_update(df_up)
        self.check_LSM_dates()

        self.LSM_obj.include_nucleocapsid_status()
        #After going through the updates, we compute the new
        #data density state of the LSM file.
        status_post = self.MPD_obj.compute_data_density(self.LSM_obj.df)
        self.MPD_obj.monotonic_increment_check(status_pre,
                status_post)

        #Uncomment the following line if you want to verify
        #that the LSM dates are consistent with the SID file.
        #self.SID_obj.check_df_dates_using_SID(self.LSM_obj.df)

        #The writing process should be executed 
        #externally for safety reasons.
        print('Do not forget to write the file to Excel.')

    def check_LSM_dates(self):
        #This function was updated on 12-Oct-2022
        #This function makes sure that all the dates
        #in the LSM file are consistent with the 
        #dates in the SID file.
        self.SID_obj.check_df_dates_using_SID(self.LSM_obj.df)

    def merge_X_with_Y_and_return_Z(self, X, Y, merge_column, kind='original+'):
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
                print(f'Unexpected {column=}')
                raise ValueError('All columns in the update should be common.')
                #If this column is new then there is no need to compare.
                #continue
            if column == merge_column:
                continue
            left  = column + '_x'
            right = column + '_y'
            #Check if we have new data
            is_new_data = Z[left].isnull() & Z[right].notnull()
            if is_new_data.any():
                print('==================START')
                print(f'{column=} has new information.')
                whats_new = Z[is_new_data]
                print(whats_new[merge_column])
                print(whats_new[right])
                print('==================END')
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
        #This function was updated on Jan 23, 2023
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
            X[self.merge_column] = np.nan
        else:
            print('The ID column already exists.')

        id_rx = re.compile('(?P<ID>[0-9]{2}[-][0-9]{7})[-][A-Z]{1,2}')

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
            vc = X[self.merge_source].value_counts()
            selection = vc.gt(1)
            print(vc[selection])
            raise ValueError('No repetitions should be present.')
        print('The merge operation is complete. Returning merged DF.')
        return X

    def generate_the_tri_sheet_file(self):
        #This function combines the Master_sans_Serology
        #the Master_avec_Serology, and the
        #Infection_column file into one Excel workbook.
        #Feb 03 2023
        fname  = 'tri_merge_may_15_2023.xlsx'
        #folder = 'Jessica_may_02_2023'
        folder = 'Tara_may_15_2023'
        fname = os.path.join('..','requests',folder, fname)
        master_avec_serology = pd.merge(self.LSM_obj.df,
                self.df, on=self.AID, how='outer')
        kind = 'Infection'
        melted_infection = self.LIS_obj.melt_infection_or_vaccination_dates(kind)

        with pd.ExcelWriter(fname) as writer:

            sh_name = 'Master_sans_Serology'
            self.df.to_excel(writer, sheet_name = sh_name, index=False)

            sh_name = 'Master_avec_Serology'
            master_avec_serology.to_excel(writer, sheet_name = sh_name, index=False)

            sh_name = 'Infection_column'
            melted_infection.to_excel(writer, sheet_name = sh_name, index=False)

            #Report data
            #sh_name = 'Report'
            #folder = self.LSM_path
            #fname = 'LSM.xlsx'
            #fname = os.path.join(folder, fname)


    def schlegel_village_update(self):
        #Use this function to update the Master file
        #when using LTC/RH data.
        #==================================================
        #Warning: Be careful with the format of the dates.
        #Use the MPD_obj.use_day_first_for_slash = True/False
        #property to set the format for each section.
        #self.MPD_obj.convert_str_to_date()
        #==================================================
        store_reformatted_update = True
        #Rename Reasons
        #for old_reason, new_reason in zip(self.MPD_obj.removal_states,
                #self.MPD_obj.new_removal_states):
            #self.df['Reason'].replace(old_reason, new_reason, inplace=True)
        fname  = 'sv_update.xlsx'
        folder = 'Tara_dec_05_2022'
        fname = os.path.join('..','requests',folder, fname)
        linf = 'Infections'
        #Read as columns of strings
        df_up = pd.read_excel(fname, dtype=str)
        df_up.replace('n/a', np.nan, inplace=True)
        df_up.replace('N/A', np.nan, inplace=True)
        #df_up.replace('refused', np.nan, inplace=True)
        #df_up.replace('Refused', np.nan, inplace=True)
        #df_up.replace('REFUSED', np.nan, inplace=True)
        #df_up.replace('DECLINED', np.nan, inplace=True)
        df_up['Reason'].replace('Refused', 'Refused-Consent', inplace=True)
        df_up['Reason'].replace('No Reconsent', 'Refused-Consent', inplace=True)
        df_up['Reason'].replace('no reconsent signed', 'Refused-Consent', inplace=True)
        df_up['Reason'].replace('Refused Consent', 'Refused-Consent', inplace=True)
        df_up['Reason'].replace('Withdrew Consent', 'Refused-Consent', inplace=True)
        txt = ('Personal history of COVID-19 – '
                'unaware of specific date as it was before he came to FV.')
        df_up.replace(txt, np.nan, inplace=True)
        df_up.replace('Refused', np.nan, inplace=True)
        df_up.replace('refused', np.nan, inplace=True)
        df_up.replace('REFUSED', np.nan, inplace=True)
        df_up.replace('None', np.nan, inplace=True)
        df_up.replace('Unknown', np.nan, inplace=True)
        df_up.replace('Yes - date unknown', np.nan, inplace=True)
        df_up.replace('COVISHEILD', 'COVISHIELD', inplace=True)
        df_up.replace(' ', np.nan, inplace=True)
        df_up.replace('n', np.nan, inplace=True)
        df_up.replace('BmodernaO', 'BModernaO', inplace=True)
        df_up.replace('Spikevax bivalent', 'BModernaO', inplace=True)
        df_up.replace('F', 'Female', inplace=True)
        df_up.replace('M', 'Male', inplace=True)
        df_up.replace('\xa0', np.nan, inplace=True)
        print(df_up)
        #self.print_column_and_datatype(df_up)
        DOB=self.MPD_obj.DOB
        DOR=self.MPD_obj.DOR
        DOE=self.MPD_obj.DOE

        #Potential irregular ID format
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


        def extract_method(txt):
            if 'PCR' in txt:
                return 'PCR'
            elif 'RAT' in txt:
                return 'RAT'
            elif 'DBS' in txt:
                return 'DBS'
            else:
                return None


        date_sep_regexp = re.compile('[ ]*[,;]+[ ]*')

        max_date_for_DOB = datetime.datetime(2000,1,1)
        dc_id_to_inf = {}
        dc_id_to_method = {}

        ID_in_ID1_or_ID2          = False
        DOR_is_merged_with_Reason = False
        find_vaccines             = True
        find_infections           = False

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
                    dor, _ = self.MPD_obj.convert_str_to_date(rpdor)
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
            if DOB in df_up.columns:
                dob = row_up[DOB]
                if pd.notnull(dob):
                    #Careful with slash format
                    #Month/Day/Year
                    dob, _ = self.MPD_obj.convert_str_to_date(dob)
                    if max_date_for_DOB < dob:
                        dob = dob - pd.DateOffset(years=100)
                        print('Removing 100 years from dob.')
                    df_up.loc[index_up, DOB] = dob
                    print(f'{dob=}')
            #=========================DOE
            if DOE in df_up.columns:
                doe = row_up[DOE]
                if pd.notnull(doe):
                    #Careful with slash format
                    #Month/Day/Year
                    doe, _ = self.MPD_obj.convert_str_to_date(doe)
                    df_up.loc[index_up, DOE] = doe
                    print(f'{doe=}')
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
                            inf, _ = self.MPD_obj.convert_str_to_date(inf)
                            L.append(inf)
                            if method:
                                M.append(method)
                    else:
                        method = extract_method(inf_str)
                        inf, _ = self.MPD_obj.convert_str_to_date(inf_str)
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
                    if pd.notnull(d_up):
                        d_up, _ = self.MPD_obj.convert_str_to_date(d_up)
                        df_up.loc[index_up, date_col] = d_up
                    if type_col in df_up.columns:
                        t_up = row_up[type_col]
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
        if DOB in df_up.columns:
            columns_with_dates.append(DOB)
        if DOR in df_up.columns:
            columns_with_dates.append(DOR)
        if DOE in df_up.columns:
            columns_with_dates.append(DOE)
        if find_vaccines:
            columns_with_dates.extend(self.LIS_obj.vaccine_date_cols)
        for column in columns_with_dates:
            df_up[column] = pd.to_datetime(df_up[column])
        self.print_column_and_datatype(df_up)
        print(df_up)

        #Replace old IDs with the new.
        self.MPD_obj.map_old_ids_to_new(df_up)

        #Date chronology
        if find_vaccines:
            #We use what is given.
            pass
            #print('Checking vaccination chronology of the update.')
            #self.LIS_obj.set_chronological_order(df_up,
                    #self.LIS_obj.vaccine_date_cols,
                    #self.LIS_obj.vaccine_type_cols,
                    #'Vaccines')


        #Storing the reformatted update.
        if store_reformatted_update:
            fname  = 'Taras_update_reformatted.xlsx'
            fname = os.path.join('..','requests',folder, fname)
            df_up.to_excel(fname, index=False)
            print(f'Wrote {fname=} to file.')

        #Merging step.
        #Be careful with the kind of update you want to execute.
        self.df = self.merge_with_M_and_return_M(df_up, 'ID', kind='original+')

        if find_vaccines:
            print('Checking vaccination chronology of the merged file.')
            self.LIS_obj.order_infections_and_vaccines()

        if find_infections:
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
                df_inf = pd.merge(df_inf, df_method, on=['ID', 'Inf #'], how='outer')
                #print(df_inf)

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


    def taras_req_jan_26_2023(self):
        #Use Nuc data to identify infections.
        #The first version was produced on January 09 2023
        #The second version was produced on January 26 2023

        HPORI = 'Has a PCR or RAT Inf'
        self.df[HPORI] = False


        DOI   = 'Date of Infection'
        LDOI   = 'Last PCR or RAT Inf'
        DOC   = self.LSM_obj.DOC

        nuc_p   = 'Has NUC+'
        self.df[nuc_p] = np.nan

        list_of_labels = ['ID', HPORI, LDOI, nuc_p]

        nuc_G = 'Nuc-IgG-100'
        nuc_A = 'Nuc-IgA-100'
        NUC   = [nuc_G, nuc_A]

        DOC_0 = DOC + '_No_Inf'
        nuc_G_0 = 'Nuc-IgG-100_No_Inf'
        nuc_A_0 = 'Nuc-IgA-100_No_Inf'
        NUC_0   = [nuc_G_0, nuc_A_0]
        nuc_G_0_s = 'Nuc-IgG-100_No_Inf_s'
        nuc_A_0_s = 'Nuc-IgA-100_No_Inf_s'
        NUC_0S   = [nuc_G_0_s, nuc_A_0_s]
        nuc_0s   = 'Nuc_No_Inf_positive'
        L0 = [DOC_0] + NUC_0 + NUC_0S + [nuc_0s]
        for x in L0:
            self.df[x] = np.nan

        list_of_labels += L0

        DOI_3 = 'Date of Infection 3mo'
        DOC_3 = DOC + '_3mo'
        days_3= 'Delta days (3mo)'
        nuc_G_3 = 'Nuc-IgG-100_3mo'
        nuc_A_3 = 'Nuc-IgA-100_3mo'
        NUC_3   = [nuc_G_3, nuc_A_3]
        nuc_G_3_s = 'Nuc-IgG-100_3mo_s'
        nuc_A_3_s = 'Nuc-IgA-100_3mo_s'
        NUC_3S   = [nuc_G_3_s, nuc_A_3_s]
        nuc_3s   = 'Nuc_3mo_positive'
        L3 = [DOI_3, DOC_3, days_3] + NUC_3 + NUC_3S + [nuc_3s]
        for x in L3:
            self.df[x] = np.nan

        list_of_labels += L3

        DOI_6 = 'Date of Infection 6mo'
        DOC_6 = DOC + '_6mo'
        days_6= 'Delta days (6mo)'
        nuc_G_6 = 'Nuc-IgG-100_6mo'
        nuc_A_6 = 'Nuc-IgA-100_6mo'
        NUC_6   = [nuc_G_6, nuc_A_6]
        nuc_G_6_s = 'Nuc-IgG-100_6mo_s'
        nuc_A_6_s = 'Nuc-IgA-100_6mo_s'
        NUC_6S   = [nuc_G_6_s, nuc_A_6_s]
        nuc_6s   = 'Nuc_6mo_positive'
        L6 = [DOI_6, DOC_6, days_6] + NUC_6 + NUC_6S + [nuc_6s]
        for x in L6:
            self.df[x] = np.nan

        list_of_labels += L6

        DOI_6p = 'Date of Infection 6+mo'
        DOC_6p = DOC + '_6+mo'
        days_6p= 'Delta days (6+mo)'
        nuc_G_6p = 'Nuc-IgG-100_6+mo'
        nuc_A_6p = 'Nuc-IgA-100_6+mo'
        NUC_6p   = [nuc_G_6p, nuc_A_6p]
        nuc_G_6p_s = 'Nuc-IgG-100_6+mo_s'
        nuc_A_6p_s = 'Nuc-IgA-100_6+mo_s'
        NUC_6pS   = [nuc_G_6p_s, nuc_A_6p_s]
        nuc_6ps   = 'Nuc_6+mo_positive'
        L6p = [DOI_6p, DOC_6p, days_6p] + NUC_6p + NUC_6pS + [nuc_6ps]
        for x in L6p:
            self.df[x] = np.nan

        list_of_labels += L6p


        nuc_G_t = 0.547779865867836
        nuc_A_t = 0.577982139779995
        nuc_t   = [nuc_G_t, nuc_A_t]

        old_date = datetime.datetime(1980,1,1)
        #Dictionaries to classify each case.
        id_to_no_inf = {}
        id_to_3mo = {}
        id_to_6mo = {}
        id_to_6mop = {}


        for index_m, row_m in self.df.iterrows():
            #Iterate over the MPD
            ID = row_m['ID']
            i_types = row_m[self.LIS_obj.positive_type_cols]
            #Select only RAT and PCR
            is_RAT_or_PCR = i_types.isin(['RAT','PCR'])
            RAT_or_PCR    = is_RAT_or_PCR[is_RAT_or_PCR]
            n_inf         = len(RAT_or_PCR)
            if n_inf == 0:
                #No infection
                self.df.loc[index_m, HPORI] = False
                selection = self.LSM_obj.df['ID'] == ID
                if ~selection.any():
                    #If no samples on record,
                    #move on to the next participant.
                    continue
                #############NO Infection case#########
                df_s = self.LSM_obj.df.loc[selection,:]
                for index_s, row_s in df_s.iterrows():
                    #Iterate over the collected samples
                    doc = row_s[DOC]
                    nuc_data = row_s[NUC]
                    if nuc_data.count() == 0:
                        #No nucleocapsid data
                        #Move on since we need this information.
                        continue
                    #Check if the NUC data is above threshold.
                    nuc_status = nuc_t < nuc_data
                    #Try to find if this ID has already stored a 
                    #NUC+ (positive) value.
                    old_sample = id_to_no_inf.get(ID, old_date)
                    if old_sample  == old_date:
                        #No True cases have been reported
                        #You may continue
                        pass
                    else:
                        #We already have a True case
                        #We only proceed if this case
                        #is also more recent and True
                        if old_sample < doc and nuc_status.any():
                            pass
                        else:
                            continue
                    self.df.loc[index_m, DOC_0] = doc
                    self.df.loc[index_m, NUC_0] = nuc_data.values
                    self.df.loc[index_m, NUC_0S] = nuc_status.values
                    if nuc_status.any():
                        id_to_no_inf[ID] = doc
                        self.df.loc[index_m, nuc_0s] = True
                        self.df.loc[index_m, nuc_p] = True
                    else:
                        self.df.loc[index_m, nuc_0s] = False

                        if pd.isnull(self.df.loc[index_m, nuc_p]):
                            self.df.loc[index_m, nuc_p] = False

                #End of for loop (samples)
                print('=====================')
                continue
            #######END OF NO Infection###############
            #######PCR/RAT+ Infection ###############
            self.df.loc[index_m, HPORI] = True
            for i_type in RAT_or_PCR.index:
                #Iterate over the infection dates
                doi_h = i_type.replace('Type','Date')
                doi   = row_m[doi_h]
                self.df.loc[index_m, LDOI] = doi
                #Samples with that ID
                selection = self.LSM_obj.df['ID'] == ID
                if ~selection.any():
                    continue
                df_s = self.LSM_obj.df.loc[selection,:]
                for index_s, row_s in df_s.iterrows():
                    #Iterate over the collected samples
                    doc = row_s[DOC]
                    #(DOI, DOC)
                    delta = (doc - doi) / np.timedelta64(1,'D')
                    if delta < 0:
                        #Collection ===> Infection
                        #The sample was not collected after the infection
                        continue
                    #Infection ===> Collection 
                    nuc_data = row_s[NUC]
                    if nuc_data.count() == 0:
                        #No nucleocapsid data
                        #Move on since we need this information.
                        continue
                    nuc_status = nuc_t < nuc_data
                    if  delta < 3*30:
                        old_doi = id_to_3mo.get(ID, old_date)
                        if old_doi  == old_date:
                            #No True cases have been reported
                            #You may continue
                            pass
                        else:
                            #We already have a True case
                            #We only proceed if this case
                            #is also more recent and True
                            if old_doi < doi and nuc_status.any():
                                pass
                            else:
                                continue
                        self.df.loc[index_m, DOI_3] = doi
                        self.df.loc[index_m, DOC_3] = doc
                        self.df.loc[index_m, days_3] = delta
                        self.df.loc[index_m, NUC_3] = nuc_data.values
                        self.df.loc[index_m, NUC_3S] = nuc_status.values
                        if nuc_status.any():
                            id_to_3mo[ID] = doi
                            self.df.loc[index_m, nuc_3s] = True
                            self.df.loc[index_m, nuc_p] = True
                        else:
                            self.df.loc[index_m, nuc_3s] = False

                            if pd.isnull(self.df.loc[index_m, nuc_p]):
                                self.df.loc[index_m, nuc_p] = False

                    elif delta < 6*30:
                        old_doi = id_to_6mo.get(ID, old_date)
                        if old_doi  == old_date:
                            #No True cases have been reported
                            #You may continue
                            pass
                        else:
                            #We already have a True case
                            #We only proceed if this case
                            #is also more recent and True
                            if old_doi < doi and nuc_status.any():
                                pass
                            else:
                                continue
                        if old_doi < doi:
                            continue

                        self.df.loc[index_m, DOI_6] = doi
                        self.df.loc[index_m, DOC_6] = doc
                        self.df.loc[index_m, days_6] = delta
                        self.df.loc[index_m, NUC_6] = nuc_data.values
                        self.df.loc[index_m, NUC_6S] = nuc_status.values
                        if nuc_status.any():
                            id_to_6mo[ID] = doi
                            self.df.loc[index_m, nuc_6s] = True
                            self.df.loc[index_m, nuc_p] = True
                        else:
                            self.df.loc[index_m, nuc_6s] = False

                            if pd.isnull(self.df.loc[index_m, nuc_p]):
                                self.df.loc[index_m, nuc_p] = False

                    #More than 6 months
                    else:
                        old_doi = id_to_6mop.get(ID, old_date)
                        if old_doi  == old_date:
                            #No True cases have been reported
                            #You may continue
                            pass
                        else:
                            #We already have a True case
                            #We only proceed if this case
                            #is also more recent and True
                            if old_doi < doi and nuc_status.any():
                                pass
                            else:
                                continue

                        self.df.loc[index_m, DOI_6p] = doi
                        self.df.loc[index_m, DOC_6p] = doc
                        self.df.loc[index_m, days_6p] = delta
                        self.df.loc[index_m, NUC_6p] = nuc_data.values
                        self.df.loc[index_m, NUC_6pS] = nuc_status.values
                        if nuc_status.any():
                            id_to_6mop[ID] = doi
                            self.df.loc[index_m, nuc_6ps] = True
                            self.df.loc[index_m, nuc_p] = True
                        else:
                            self.df.loc[index_m, nuc_6ps] = False

                            if pd.isnull(self.df.loc[index_m, nuc_p]):
                                self.df.loc[index_m, nuc_p] = False

            print('=====================')
        fname = 'Nuc_inf_classification.xlsx'
        folder= 'Tara_jan_26_2023'
        fname = os.path.join(self.requests_path, folder, fname)
        df = self.df[list_of_labels]
        df.to_excel(fname, index=False)


    def taras_req_2_jan_26_2023(self):
        #Use Nuc data to identify infections.
        #This file creates the 2X2 Nuc/Inf table.
        fname = 'Nuc_inf_classification.xlsx'
        folder= 'Tara_jan_26_2023'
        fname = os.path.join(self.requests_path, folder, fname)
        df = pd.read_excel(fname)

        HPORI = 'Has a PCR or RAT Inf'


        DOI   = 'Date of Infection'
        LDOI   = 'Last PCR or RAT Inf'
        DOC   = self.LSM_obj.DOC

        nuc_p   = 'Has NUC+'

        list_of_labels = ['ID', HPORI, LDOI, nuc_p]

        nuc_G = 'Nuc-IgG-100'
        nuc_A = 'Nuc-IgA-100'
        NUC   = [nuc_G, nuc_A]

        DOC_0 = DOC + '_No_Inf'
        nuc_G_0 = 'Nuc-IgG-100_No_Inf'
        nuc_A_0 = 'Nuc-IgA-100_No_Inf'
        NUC_0   = [nuc_G_0, nuc_A_0]
        nuc_G_0_s = 'Nuc-IgG-100_No_Inf_s'
        nuc_A_0_s = 'Nuc-IgA-100_No_Inf_s'
        NUC_0S   = [nuc_G_0_s, nuc_A_0_s]
        nuc_0s   = 'Nuc_No_Inf_positive'
        L0 = [DOC_0] + NUC_0 + NUC_0S + [nuc_0s]

        list_of_labels += L0

        DOI_3 = 'Date of Infection 3mo'
        DOC_3 = DOC + '_3mo'
        days_3= 'Delta days (3mo)'
        nuc_G_3 = 'Nuc-IgG-100_3mo'
        nuc_A_3 = 'Nuc-IgA-100_3mo'
        NUC_3   = [nuc_G_3, nuc_A_3]
        nuc_G_3_s = 'Nuc-IgG-100_3mo_s'
        nuc_A_3_s = 'Nuc-IgA-100_3mo_s'
        NUC_3S   = [nuc_G_3_s, nuc_A_3_s]
        nuc_3s   = 'Nuc_3mo_positive'
        L3 = [DOI_3, DOC_3, days_3] + NUC_3 + NUC_3S + [nuc_3s]

        list_of_labels += L3

        DOI_6 = 'Date of Infection 6mo'
        DOC_6 = DOC + '_6mo'
        days_6= 'Delta days (6mo)'
        nuc_G_6 = 'Nuc-IgG-100_6mo'
        nuc_A_6 = 'Nuc-IgA-100_6mo'
        NUC_6   = [nuc_G_6, nuc_A_6]
        nuc_G_6_s = 'Nuc-IgG-100_6mo_s'
        nuc_A_6_s = 'Nuc-IgA-100_6mo_s'
        NUC_6S   = [nuc_G_6_s, nuc_A_6_s]
        nuc_6s   = 'Nuc_6mo_positive'
        L6 = [DOI_6, DOC_6, days_6] + NUC_6 + NUC_6S + [nuc_6s]

        list_of_labels += L6

        DOI_6p = 'Date of Infection 6+mo'
        DOC_6p = DOC + '_6+mo'
        days_6p= 'Delta days (6+mo)'
        nuc_G_6p = 'Nuc-IgG-100_6+mo'
        nuc_A_6p = 'Nuc-IgA-100_6+mo'
        NUC_6p   = [nuc_G_6p, nuc_A_6p]
        nuc_G_6p_s = 'Nuc-IgG-100_6+mo_s'
        nuc_A_6p_s = 'Nuc-IgA-100_6+mo_s'
        NUC_6pS   = [nuc_G_6p_s, nuc_A_6p_s]
        nuc_6ps   = 'Nuc_6+mo_positive'
        L6p = [DOI_6p, DOC_6p, days_6p] + NUC_6p + NUC_6pS + [nuc_6ps]

        list_of_labels += L6p

        fname = 'Nuc_inf_matrices.xlsx'
        folder= 'Tara_jan_26_2023'
        fname = os.path.join(self.requests_path, folder, fname)

        nuc_array = [np.array(['Nuc', 'Nuc']), np.array(['+', '-'])]
        inf_array = [np.array(['Inf', 'Inf']), np.array(['+', '-'])]
        m = pd.DataFrame(np.zeros((2,2)), index = nuc_array, columns=inf_array)

        nuc_list = [nuc_3s, nuc_6s, nuc_6ps]

        with pd.ExcelWriter(fname) as writer:
            for nuc_xs in nuc_list:
                #Positive Infection and positive Nuc 3mo
                s1 = df[HPORI] == True
                s2 = df[nuc_xs] == True
                df_s = df[s1 & s2]
                nuc_pos_inf_pos = len(df_s)
                m.loc[('Nuc','+'),('Inf','+')] = nuc_pos_inf_pos

                #Positive Infection and negative Nuc 3mo
                s1 = df[HPORI] == True
                s2 = df[nuc_xs] == False
                df_s = df[s1 & s2]
                nuc_neg_inf_pos = len(df_s)
                m.loc[('Nuc','-'),('Inf','+')] = nuc_neg_inf_pos

                #Negative Infection and positive Nuc 
                s1 = df[HPORI] == False
                s2 = df[nuc_0s] == True
                df_s = df[s1 & s2]
                nuc_pos_inf_neg = len(df_s)
                m.loc[('Nuc','+'),('Inf','-')] = nuc_pos_inf_neg

                #Negative Infection and negative Nuc 3mo
                s1 = df[HPORI] == False
                s2 = df[nuc_0s] == False
                df_s = df[s1 & s2]
                nuc_neg_inf_neg = len(df_s)
                m.loc[('Nuc','-'),('Inf','-')] = nuc_neg_inf_neg

                sh_name = nuc_xs
                m.to_excel(writer, sheet_name = sh_name)


    def extract_ID_from_sv_file(self):
        #Feb 07 2023
        #Extract the ID from Deanna's file
        #There are two columns, and whenever it is not 
        #possible to extract the ID from the first, we
        #try with the second one.
        fname = 'sv_data_feb_6_2023.xlsx'
        folder= 'Deanna_feb_16_2023'
        fname = os.path.join(self.requests_path, folder, fname)
        df_up = pd.read_excel(fname)
        df_up['ID'] = df_up['ID'].str.replace(' ','')
        df_up['Barcode'] = df_up['Barcode'].str.replace(' ','')

        df_up.replace('n/a', np.nan, inplace=True)
        df_up.replace('N/A', np.nan, inplace=True)
        df_up.replace('refused', np.nan, inplace=True)
        df_up.replace('Refused', np.nan, inplace=True)
        df_up.replace('REFUSED', np.nan, inplace=True)
        df_up.replace('DECLINED', np.nan, inplace=True)
        print(df_up)

        v_slice = slice('Vaccine Date 1', 'Vaccine Type 5')
        rexp = re.compile('[0-9]{2}-[0-9]{7}')
        L = []
        n_new_vaccine_dates = 0
        for index_up, row_up in df_up.iterrows():

            ID_t = row_up['ID']
            bc_t = row_up['Barcode']
            flag_found_ID = False
            index_m = -1

            if pd.notnull(ID_t):
                obj = rexp.search(ID_t)
                if obj:
                    ID = obj.group(0)
                    selector = self.df['ID'] == ID
                    if selector.any():
                        index_m = selector[selector].index[0]
                        L.append(ID)
                        flag_found_ID = True
                        #print(ID)


            if (not flag_found_ID) and pd.notnull(bc_t):
                obj = rexp.search(bc_t)
                if obj:
                    ID = obj.group(0)
                    selector = self.df['ID'] == ID
                    if selector.any():
                        index_m = selector[selector].index[0]
                        L.append(ID)
                        flag_found_ID = True
                        #print(ID)

            if not flag_found_ID:
                raise ValueError('No ID')

            #Vaccine time
            print(f'============{ID=}=======')
            v_items = row_up[v_slice]
            for col_name, col_value in v_items.items():
                if pd.notnull(col_value):
                    print(f'Working with: {col_name=}')
                    stored = self.df.loc[index_m, col_name]
                    if pd.notnull(stored):
                        if stored != col_value:
                            print('We have a difference')
                            print(f'{stored=}')
                            print(f'{col_value=}')
                        else:
                            pass
                    else:
                        print('Storing:')
                        print(f'{col_value=}')
                        if 'Type' in col_name:
                            if col_value in self.LIS_obj.list_of_valid_vaccines:
                                self.df.loc[index_m, col_name] = col_value
                            else:
                                raise ValueError('Vaccine type undefined')
                        elif 'Date' in col_name:
                            if 'time' in str(type(col_value)):
                                self.df.loc[index_m, col_name] = col_value
                                n_new_vaccine_dates += 1
                            else:
                                raise ValueError('Vaccine date undefined')




        print(f'{n_new_vaccine_dates=}')
        print(len(L))
        #df = pd.DataFrame({'ID':L})
        #fname = 'ids.xlsx'
        #folder= 'Lindsay_feb_07_2023'
        #fname = os.path.join(self.requests_path, folder, fname)
        #df.to_excel(fname, index=False)




    def create_raw_files_for_template(self):
        #Feb 07 2023
        #This function creates a spreadsheet that
        #can be used to populate the template file
        #for the retirement and LTC resident homes.


        #self.MPD_obj.add_site_column(self.df)
        self.print_column_and_datatype(self.df)
        #=============================================
        #These lines eliminate the non-date cells.
        #=============================================
        #for index_m, row_m in self.df.iterrows():
            #print('=====================')
            #ID = row_m['ID']
            #print(ID)
            #for col, code in self.SID_obj.i_blood_draw_code_to_col_name.items():
                #data = row_m[col]
                #if pd.isnull(data):
                    #continue
                #dtype = str(type(data))
                #if 'time' in dtype:
                    #pass
                #else:
                    #print('Erasing:', data)
                    #self.df.loc[index_m, col] = np.nan
        fname = 'template_h.xlsx'
        folder= 'Deanna_feb_17_2023'
        fname = os.path.join(self.requests_path, folder, fname)
        df_r = pd.read_excel(fname, sheet_name = 'Rel')
        df_h = pd.read_excel(fname, sheet_name = 'Report')
        print(df_r)
        print(df_h)

        source_to_target  = {}
        anticipated_dc    = {}

        #===============================
        #Select the sites (If necessary)
        #===============================
        #sites = [20,61]
        sites = [2, 3, 5, 6, 7,
                9, 11, 12, 14, 19,
                50, 51, 52, 53, 54, 55, 56]
        selector = pd.isnull(self.df['ID'])
        for site in sites:
            selector |= self.df['Site'] == site

        #================================================
        #Source data frame to be converted into a raw file
        #================================================
        #fname = 'ids.xlsx'
        #fname = os.path.join(self.requests_path, folder, fname)
        #df_ids= pd.read_excel(fname)
        #selector = self.df['ID'].isin(df_ids['ID'])


        df_s = self.df[selector].copy()

        #Extract the new names for the source df
        for index, row in df_r.iterrows():

            source = row['Source']
            target = row['Target']
            #Target example: Blood Draw - 3 Week Post 1st (Anticipated)
            aux    = row['Aux']
            add    = row['Add']

            if pd.notnull(source):
                source_to_target[source] = target

            if pd.notnull(aux):
                aux_with_new_name = source_to_target[aux]
                anticipated_dc[target] = (aux_with_new_name, add)

        df_s.rename(columns=source_to_target, inplace=True)

        #Add new columns to the source df
        for column in df_h.columns:
            if column not in df_s.columns:
                #print(column)
                df_s[column] = np.nan


        #Store the names of the target df
        original_columns = df_h.columns

        #Note that due to the concatenation
        #the headers use the new names.
        df_m = pd.concat([df_h, df_s], axis=0, join='inner')
        df_m = df_m[original_columns].copy()

        #Create barcode
        #Because the barcode does not have a predictable
        #pattern, we do not include it into the template.
        #def create_barcode(txt):
            #return 'LTC1-' + txt

        #df_m['Barcode'] = df_m['ID'].apply(create_barcode)

        #self.print_column_and_datatype(df_m)



        #Add anticipated dates
        v_dates_t = self.LIS_obj.vaccine_date_cols
        v_dates_h = []
        for v_date in v_dates_t:
            v_dates_h.append(source_to_target[v_date])

        REF = 'Study Refusals'
        rexp = re.compile('(?P<number>[0-9]+)[a-z]+')

        def extract_number(txt):
            #For example, if we have the text "4th Dose"
            #we want to extract the 4.
            obj = rexp.search(txt)
            if obj:
                return int(obj.group('number'))

        #================
        #================Iterate over all participants in df_m
        #================
        for index, row in df_m.iterrows():
            #if row['ID'] == '02-1912375':
                #return
            print(row['ID'])

            #Refusal
            if pd.notnull(row[REF]):
                ref = row[REF]
                if 'blood' in ref.lower():
                    print('Changing:', ref, ' to B:Blood')
                    df_m.loc[index,REF] = 'B: Blood'
                    continue
                else:
                    print('Erasing:', ref)
                    df_m.loc[index,REF] = np.nan

            #Reason
            if pd.notnull(row['Reason']):
                print('Not active')
                continue


            v_dates = row[v_dates_h]
            #No vaccines
            if v_dates.count() == 0:
                print('No vaccines')
                continue

            selector = v_dates.notnull()
            v_dates = v_dates[selector]
            print(v_dates)
            last_vaccine = extract_number(v_dates.index[-1])

            for target, (aux, offset) in anticipated_dc.items():

                #Target example: Blood Draw - 3 Week Post 1st (Anticipated)
                #We want to extract the "1" from "1st".

                if pd.isnull(row[aux]):
                    continue

                post_dose = extract_number(target)

                if post_dose < last_vaccine:
                    #If the last vaccine took place after 
                    #the "post_dose-th" = "1st", then ignore.
                    print(post_dose, ' < ', last_vaccine)
                    continue

                actual = target.replace('Anticipated', 'Actual')
                #Actual example: Blood Draw - 3 Week Post 1st (Actual)
                if pd.notnull(row[actual]):
                    #If we have it, then why compute an anticipated date?
                    print(actual, ' has ', row[actual])
                    continue

                #If we are still here, add the offset.
                df_m.loc[index,target] = row[aux] + pd.DateOffset(days=offset)

        #for target, (aux, offset) in anticipated_dc.items():
            ##If the vaccine column is empty, ignore.
            #if df_m[aux].isnull().all():
                #continue
            #df_m[target] = df_m[aux] + pd.DateOffset(days=offset)
            #print('-------------')

        #Write dates in format day-Month-Year
        #Don't forget to also change the blood draw columns.
        for col, col_type in zip(df_m.columns,df_m.dtypes):
            print('===============',col)
            if 'time' in str(col_type):
                print('>>>>>Time', col, col_type)
                df_m[col] = df_m[col].dt.strftime('%d-%b-%Y')
                continue
            if 'Blood' in col:
                if df_m[col].count() == 0:
                    continue
                print('>>>>>Blood', col, col_type)
                df_m[col] = pd.to_datetime(df_m[col])
                df_m[col] = df_m[col].dt.strftime('%d-%b-%Y')
                continue

        #Zero-pad the site column to 2 digits
        df_m['Site'] = df_m['Site'].astype(str).str.zfill(2)

        fname = 'raw_data_list_feb_17_2023.xlsx'
        fname = os.path.join(self.requests_path, folder, fname)
        df_m.to_excel(fname, index=False)



    def ahmads_request_feb_07_2023(self):
        #Feb 07 2023
        #Get the full IDs of the samples that have to be processed.
        fname = 'missing_data.xlsx'
        folder= 'Ahmad_feb_06_2023'
        fname = os.path.join(self.requests_path, folder, fname)
        with pd.ExcelWriter(fname) as writer:
            for col in self.LSM_obj.df.columns:
                if col == 'ID':
                    continue
                if col == 'Full ID':
                    continue
                if col == 'Date collected':
                    continue
                selector = self.LSM_obj.df[col].isnull()
                samples = self.LSM_obj.df.loc[selector,'Full ID']
                samples.to_excel(writer, sheet_name = col, index = False)

    def deannas_request_feb_14_2023(self):
        #Feb 07 2023
        #Extract the ID from Deanna's file
        #There are two columns, and whenever it is not 
        #possible to extract the ID from the first, we
        #try with the second one.
        folder= 'Deanna_feb_14_2023'
        fname = 'ids.xlsx'
        fname = os.path.join(self.requests_path, folder, fname)
        df = pd.read_excel(fname)
        df['ID_unique'] = np.nan
        df['ID_repeated'] = np.nan
        vc = df['ID'].value_counts()
        vc = vc[vc.gt(1)]
        vc = vc.index.to_list()
        df.iloc[:len(vc),2] = vc
        unique = df['ID'].unique()
        df.iloc[:len(unique),1] = unique
        fname = 'ids_with_decomposition.xlsx'
        fname = os.path.join(self.requests_path, folder, fname)
        df.to_excel(fname, index=False)

    def taras_request_feb_14_2023(self):
        #Terminate individuals from 15,16,17 using last blood draw.
        s = self.df['ID'].isnull()
        sites = [15,16,17]
        for site in sites:
            s |= self.df['Site'] == site
        blood_slice = slice('Blood Draw:Baseline - B','Blood Draw:Repeat - JR')
        for index, row in self.df.loc[s,:].iterrows():
            blood_dates = row[blood_slice]
            selector = blood_dates.notnull()
            blood_dates = blood_dates[selector]
            if len(blood_dates) == 0:
                print('Empty')
            else:
                date = blood_dates.max()
                ID = row['ID']
                self.df.loc[index, 'Reason'] = 'Withdrew'
                self.df.loc[index, 'Date Removed from Study'] = date
                print('Terminated ', ID, ' on the ', date)

    def taras_request_feb_16_2023(self):
        #Use Tara's file to update the activation status.
        folder= 'Tara_feb_16_2023'
        fname = 'update.xlsx'
        fname = os.path.join(self.requests_path, folder, fname)
        df_up = pd.read_excel(fname, usecols = 'A:I',
                sheet_name = 'Master_sans_Serology')
        df_up.dropna(subset='Active', inplace=True)
        #print(df_up['Active'])
        DOR = 'Date Removed from Study'
        n_active_changes = 0
        for index_up, row_up in df_up.iterrows():
            status_up = row_up['Active']
            ID = row_up['ID']
            print(f'{ID=}')
            selector = self.df['ID'] == ID
            index_m = selector[selector].index[0]
            status_m = self.df.loc[index_m,'Active']
            if pd.notnull(status_up):
                if status_up == 1:
                    if status_m == True:
                        #print('Equal')
                        pass
                    else:
                        print('>>>>Different')
                        self.df.loc[index_m, 'Active'] = True
                        self.df.loc[index_m, 'Reason'] = np.nan
                        self.df.loc[index_m,  DOR] = np.nan
                        n_active_changes += 1

        print(f'{n_active_changes=}')


    def taras_request_feb_17_2023(self):
        #Extract the ID from Deanna's file
        #There are two columns, and whenever it is not 
        #possible to extract the ID from the first, we
        #try with the second one.
        fname = 'update_62.xlsx'
        folder= 'Tara_feb_17_2023'
        fname = os.path.join(self.requests_path, folder, fname)

        inf_date_h = self.LIS_obj.positive_date_cols

        df_up = pd.read_excel(fname, dtype={'Infections':str})
        df_up['ID'] = df_up['ID'].str.replace(' ','')

        df_up.replace('n/a', np.nan, inplace=True)
        df_up.replace('N/A', np.nan, inplace=True)
        df_up.replace('refused', np.nan, inplace=True)
        df_up.replace('Refused', np.nan, inplace=True)
        df_up.replace('REFUSED', np.nan, inplace=True)
        df_up.replace('DECLINED', np.nan, inplace=True)
        df_up.replace('BmodernaO', 'BModernaO', inplace=True)
        df_up.replace('moderna', 'Moderna', inplace=True)
        df_up.replace('pfizer', 'Pfizer', inplace=True)
        print(df_up)

        v_slice = slice('Vaccine Date 1', 'Vaccine Type 5')
        dor_rexp = re.compile('[a-zA-Z]+[ ][0-9]+.*')
        L = []
        n_new_vaccine_dates = 0
        for index_up, row_up in df_up.iterrows():

            ID = row_up['ID']
            selector = self.df['ID'] == ID
            flag_found_ID = False

            if selector.any():
                flag_found_ID = True

            if not flag_found_ID:
                raise ValueError('No ID')

            index_m = selector[selector].index[0]

            #Vaccine time
            print(f'============{ID=}=======')
            v_items = row_up[v_slice]
            for col_name, col_value in v_items.items():
                if pd.notnull(col_value):
                    print(f'Working with: {col_name=}')
                    stored = self.df.loc[index_m, col_name]
                    if pd.notnull(stored):
                        if stored != col_value:
                            print('We have a difference')
                            print(f'{stored=}')
                            print(f'{col_value=}')
                        else:
                            pass
                    else:
                        print('Storing:')
                        print(f'{col_value=}')
                        if 'Type' in col_name:
                            if col_value in self.LIS_obj.list_of_valid_vaccines:
                                self.df.loc[index_m, col_name] = col_value
                            else:
                                raise ValueError('Vaccine type undefined')
                        elif 'Date' in col_name:
                            if 'time' in str(type(col_value)):
                                self.df.loc[index_m, col_name] = col_value
                                n_new_vaccine_dates += 1
                            else:
                                raise ValueError('Vaccine date undefined')

            #Infection time
            infections = row_up['Infections']
            if pd.notnull(infections):
                if ',' in infections:
                    L = infections.split(',')
                else:
                    L = [infections]
                for txt in L:
                    date_up = pd.to_datetime(txt, dayfirst=False)
                    print('>>>>>>>>>', date_up)
                    inf_dates = self.df.loc[index_m, inf_date_h]
                    for col_name, inf_date in inf_dates.items():
                        if pd.notnull(inf_date):
                            delta = (inf_date - date_up) / np.timedelta64(1,'D')
                            if np.abs(delta) < 7:
                                print('We have it.')
                                break
                        else:
                            raise ValueError('Is new?')

            #Reasons time
            reason_plus_date = row_up['Date Removed from Study']
            if pd.notnull(reason_plus_date):
                flag_found_reason = False
                for k,reason in enumerate(self.MPD_obj.removal_states_l):
                    if reason in reason_plus_date.lower():
                        reason_up = self.MPD_obj.removal_states[k]
                        flag_found_reason = True
                        break

                if not flag_found_reason:
                    print(reason_plus_date)
                    raise ValueError('Unknown reason')

                obj = dor_rexp.search(reason_plus_date)
                if obj:
                    txt = obj.group(0)
                    dor_up = pd.to_datetime(txt, dayfirst = False)
                else:
                    raise ValueError('Unknown DOR')

                reason_m = self.df.loc[index_m, 'Reason']
                dor_m = self.df.loc[index_m, self.MPD_obj.DOR]

                if pd.notnull(reason_m):
                    if reason_m != reason_up:
                        print('++++++++++Different reason)')
                        print('Original:', reason_m)
                        print('New     :', reason_up)
                        print('Updating reason')
                        self.df.loc[index_m, 'Reason'] = reason_up
                else:
                    print('Storing reason:', reason_up)
                    self.df.loc[index_m, 'Reason'] = reason_up

                if pd.notnull(dor_m):
                    if dor_m != dor_up:
                        print('++++++++++Different dor)')
                        print('Original:', dor_m)
                        print('New     :', dor_up)
                        print('Updating dor')
                        self.df.loc[index_m, self.MPD_obj.DOR] = dor_up
                else:
                    print('Storing DOR:', dor_up)
                    self.df.loc[index_m, self.MPD_obj.DOR] = dor_up


        print(f'{n_new_vaccine_dates=}')



    def taras_request_feb_27_2023(self):
        #Merge Zain's file with the MPD file.
        fname = 'zains_data.xlsx'
        folder= 'Tara_feb_27_2023'
        fname = os.path.join(self.requests_path, folder, fname)
        df_z = pd.read_excel(fname)
        if df_z.value_counts().gt(1).any():
            raise ValueError('Repeats')
        df_z = df_z[['ID', 'Frailty Scale']]
        df_z['Frailty Scale'] = df_z['Frailty Scale'].str.replace('CFS','')
        df_z['Frailty Scale'] = df_z['Frailty Scale'].str.replace(' ','')
        df_z['Frailty Scale'] = df_z['Frailty Scale'].astype(float)
        df_z = pd.merge(self.df, df_z, on='ID', how='outer')
        df_z['Frailty Scale_x'] = df_z['Frailty Scale_y']
        df_z.drop(columns='Frailty Scale_y', inplace=True)
        df_z.rename(columns={'Frailty Scale_x':'Frailty Scale'}, inplace=True)
        self.print_column_and_datatype(df_z)
        self.df = df_z.copy()












obj = Merger()
#obj.REP_obj.generate_individual_PDFs()
#obj.REP_obj.find_repetitions_in_tri_survey()
#obj.REP_obj.manage_moves()
#obj.REP_obj.generate_medication_groups()
#obj.MPD_obj.update_active_status_column()
#obj.MPD_obj.compute_age_from_dob()
#obj.LIS_obj.update_PCR_and_infection_status()
#obj.write_the_M_file_to_excel()
#obj.LSM_obj.check_vaccine_labels()
#obj.LIS_obj.generate_list_of_missing_vaccine_types()
obj.generate_the_tri_sheet_file()

#obj.MPD_obj.clean_DBS_from_file()
#obj.MPD_obj.contrast_template_with_M_file_V2()

#obj.SID_obj.find_repeated_dates_in_megans_file()
#obj.SID_obj.update_master_using_SID_V2()
#obj.write_the_M_file_to_excel()

#obj.MPD_obj.peace_of_mind_check()
#obj.LIS_obj.order_infections_and_vaccines()
