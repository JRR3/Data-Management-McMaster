#JRR @ McMaster University
#Update: 18-Nov-2022
import numpy as np
import pandas as pd
import os
import re
import matplotlib as mpl
mpl.rcParams['figure.dpi']=300
import matplotlib.pyplot as plt
import datetime


# <b> Section: Master_Participant_Data </b>

class MasterParticipantData:
    def __init__(self, path, parent=None):

        self.parent       = None
        self.delta_report = None

        self.merge_column = 'ID'
        self.DOR      = 'Date Removed from Study'
        self.DOE      = 'Enrollment Date'
        #Date of birth
        self.DOB      = 'DOB'
        self.reason   = 'Reason'
        self.is_active   = 'Active'
        self.site_type = 'LTC or RH'
        self.comments = 'Notes/Comments'
        #self.note     = 'Notes/Comments'
        #Moved Out --> Moved
        #self.old_removal_states = ['Deceased', 'Discharged', 'Moved',
                               #'Decline', 'Withdraw from Study',
                               #'Withdrew Consent']
        self.RC = 'Refused-Consent'
        self.removal_states = ['Deceased', 'Discharged', 'Moved',
                               'Declined', 'Withdrew', self.RC]
        self.removal_states_l = [x.lower() for x in self.removal_states]

        self.yearfirst_regexp = re.compile('[0-9]{4}[-][0-9]{2}[-][0-9]{2}')

        self.dayfirst_regexp = re.compile('[0-9]+[-][a-zA-Z]+[-](?P<year>[0-9]+)')

        self.dayfirst_with_slash_regexp = re.compile('[0-9]+[/][0-9]+[/](?P<year>[0-9]+)')

        self.use_day_first_for_slash = False
        txt = ('(?P<month>[0-9]+)' + '[/]' +
        '(?P<day>[0-9]+)' + '[/]' +
        '(?P<year>[0-9]+)')
        self.monthfirst_regexp = re.compile(txt)

        txt = ('(?P<month>[a-zA-Z]+)' + '[ ]+' +
        '(?P<day>[0-9]{1,2})' + '[a-z]*' +
        '[,]?' + '[ ]*' +
        '(?P<year>[0-9]{2,})')
        self.monthfirst_as_text_regexp = re.compile(txt)

        if parent:
            self.parent = parent
            print('MPD class initialization from Manager.')
        else:
            #Which row contains the first data entry in the Excel file
            self.excel_starts_at = 3
            self.dpath = path
            MPD = 'Master_Participant_Data'
            self.fname = os.path.join(self.dpath, MPD + '.xlsx')
            #Read the Excel file containing the data
            self.parent.df = pd.read_excel(self.fname,
                                    sheet_name="Master", skiprows=[0])



    def check_for_repeats(self):
        #We drop rows containing NA in the "Sample ID" from the
        #file because sometimes there are additional comments
        #below the table that get interpreted as additional rows.
        #self.parent.df.dropna(subset=[self.merge_column], inplace=True)
        print('>>>Making sure there are no repeats')
        value_count_gt_1 = self.parent.df[self.merge_column].value_counts().gt(1)
        if value_count_gt_1.any():
            print('We have repetitions in Master Participant Data')
            print('Check the column:', self.merge_column)
            raise Exception('No repetitions were expected.')
        else:
            print('No repeats were found in Master Participant Data')



    def add_Y_in_the_Whole_Blood_column(self):
        fname = 'should_have_a_y_under_whole_blood.xlsx'
        txt = os.path.join(self.dpath, fname)
        df_ids = pd.read_excel(txt)
        up_col = 'Whole Blood (Y,N,N/A)'
        for _, update_row in df_ids.iterrows():
            ID = update_row['ID']
            state = update_row['Whole Blood']
            #Note that we are requesting ALL the rows that have the
            #given ID. However, in practive we expect to get only one.
            rows = self.parent.df[self.parent.df['ID']== ID]
            for index, row in rows.iterrows():
                if row[up_col] != 'Y':
                    self.parent.df.loc[index, up_col] = state
                    print('Made a change in the WB column for', ID)
                    print('Changed to', state)



    ##########Sep 22 2022##################

    def update_reason_dates_and_status(self, df_up):
        print('Update the Reason, the DOR and the status (Active).')
        for index, row in df_up.iterrows():
            ID = row['ID']
            date = row['date']
            reason = row[self.reason]
            selector = self.parent.df['ID'] == ID
            if not selector.any():
                raise ValueError('ID does not exist.')
            self.parent.df.loc[selector, self.DOR] = date
            self.parent.df.loc[selector, self.reason] = reason

        print('The DOR and Reason columns have been updated.')

        self.update_active_status_column()


    def update_active_status_column(self):
        c1 = self.parent.df[self.DOR].notnull()
        c2 = self.parent.df[self.reason].notnull()
        self.parent.df[self.is_active] = ~(c1 | c2)
        print('The active status column has been updated.')
        print('Do not forget to write the df to Excel.')


    def add_site_column(self, df):
        #Feb 03 2023
        #Note that this flag has to be active for the 
        #function:
        #melt_infection_or_vaccination_dates(self, kind='Infection')
        #within the LIS class.
        distinguish_between_LTC_and_RH = True
        rexp = re.compile('(?P<site>[0-9]+)[-](?P<user>[0-9]+)')
        def get_site(txt):
            obj = rexp.search(txt)
            if obj:
                return int(obj.group('site'))
            else:
                raise ValueError('Unable to extract site')

        site = 'Site'
        if site in df.columns:
            pass
        else:
            df[site] = df['ID']
            df[site] = df[site].apply(get_site)
        #If we need to distinguish between LTC and RH
        #proceed to the next instructions.
        if distinguish_between_LTC_and_RH:
            df[self.site_type] = 'LTC'
            df[self.site_type] = df[self.site_type].where(df[site] < 50, 'RH')

        sl = slice('Sex','Blood Draw:Repeat - JR')
        cols  = ['ID', site]
        cols += df.loc[:,sl].columns.to_list()
        self.parent.df = df[cols].copy()

    ##########Oct 19 2022##################
    def update_M_from_comments_and_dates(self):
        #Example
        #ID          COMMENTS            DOR
        #05-2924993  XYZ+DISCHARGED+XYZ  9/20/2022
        #Be careful with the date formats.
        #For this data set the first 
        #number represents the month.
        fname = 'data.xlsx'
        folder = 'Megan_2_19_oct_2022'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df_up = pd.read_excel(fname)
        print(df_up)
        L = []
        for index, row in df_up.iterrows():
            comment = row['COMMENTS']
            for k,reason in enumerate(self.removal_states_l):
                if reason in comment.lower():
                    R = self.removal_states[k]
                    L.append(R)
                    break
            date = row[self.DOR]
            print(date)
            if isinstance(date, str):
                print('String')
                dt = pd.to_datetime(date, dayfirst=False)
                print(f'{dt=}')
                print('===============')
                df_up.loc[index, self.DOR] = dt
            elif isinstance(date, datetime.datetime):
                print('Date')
                pass
            else:
                raise ValueError('Unexpected type')

        df_up['Reason'] = L
        df_up[self.DOR] = pd.to_datetime(df_up[self.DOR])
        self.parent.print_column_and_datatype(df_up)
        cols_to_keep = ['ID', self.DOR, 'Reason']
        df_up = df_up[cols_to_keep]
        print(df_up)
        self.parent.df = self.parent.merge_with_M_and_return_M(df_up, 'ID', kind='original+')
        print('Ready to write M file to Excel.')
        self.parent.write_the_M_file_to_excel()


    ##########Jan 09 2023##################
    def compute_age_from_dob(self, df):
        for index, row in df.iterrows():
            dob = row['DOB']
            if pd.isnull(dob):
                continue
            today = datetime.datetime.now()
            delta = (today - dob).days
            years = delta // 365
            df.loc[index,'Age'] = years

    ##########Oct 25 2022##################
    def missing_DOR(self):
        #Missing dates of removal, i.e.,
        #Date Removed from Study
        not_active_s = self.parent.df[self.is_active] == False
        missing_date_s = self.parent.df[self.DOR].isnull()
        selector = not_active_s & missing_date_s
        S = self.parent.df['ID'].where(selector, np.nan)
        selector = S.notnull()
        S = S.loc[selector]
        print(S)

    ##########Nov 18 2022##################
    def short_year_to_long_year(self, obj):
        #22 --> 2022
        #23 --> 1923
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


    def convert_str_to_date(self, txt, use_day_first_for_slash=None):
        #Note that we are using an external variable to define
        #the behavior.
        if use_day_first_for_slash is None:
            use_day_first_for_slash = self.use_day_first_for_slash
        if pd.isnull(txt):
            raise ValueError('Object is NAN')
        obj = self.monthfirst_as_text_regexp.search(txt)
        if obj:
            #Month(text)/Day/Year
            #Check if we have a short year.
            date = self.short_year_to_long_year(obj)
            date = pd.to_datetime(date, dayfirst=False, yearfirst=False)
        elif '/' in txt:
            if use_day_first_for_slash:
                obj = self.dayfirst_with_slash_regexp.search(txt)
                if obj:
                    date = obj.group(0)
                    date = pd.to_datetime(date, dayfirst=True)
                else:
                    print(txt)
                    raise ValueError('Unexpected date for slash with day first.')
            else:
                #Month(number)/Day/Year
                obj = self.monthfirst_regexp.search(txt)
                if obj:
                    #Check if we have a short year.
                    date = self.short_year_to_long_year(obj)
                    month_str = obj.group('month')
                    month_int = int(month_str)
                    if 12 < month_int:
                        print('Unexpected format: Month/Day/Year but Month > 12')
                        date = pd.to_datetime(date, dayfirst=True)
                    else:
                        date = pd.to_datetime(date, dayfirst=False, yearfirst=False)
                else:
                    print(f'{txt=}')
                    raise ValueError('Unknown format for date.')
        else:
            #In this case we do not expect a short year.
            obj = self.yearfirst_regexp.search(txt)
            if obj:
                date = obj.group(0)
                date = pd.to_datetime(date, yearfirst=True)
            else:
                obj = self.dayfirst_regexp.search(txt)
                if obj:
                    #Check if we have a short year.
                    #date = obj.group(0)
                    date = self.short_year_to_long_year(obj)
                    date = pd.to_datetime(date, dayfirst=True)
                else:
                    print(f'{txt=}')
                    raise ValueError('Unknown format for date.')
        return (date, obj.group(0))


    def stratification_by_inf_and_vac(self):
        #Tara requested this function on Nov 24 2022.
        #Updated on Nov 30 2022
        #Generate_7_tables
        #Load the full master file.
        print('Warning: Make sure the W file is updated.')
        fname  = 'W.xlsx'
        fname = os.path.join(self.parent.outputs_path, fname)
        df = pd.read_excel(fname)
        DOC = self.parent.LSM_obj.DOC

        v_date_cols = self.parent.LIS_obj.vaccine_date_cols
        inf_date_cols = self.parent.LIS_obj.positive_date_cols
        #Iterate over the S+M data frame and look for 'G'
        candidates = 0
        case_to_list = {}
        case_to_list[1] = []
        case_to_list[2] = []
        case_to_list[3] = []
        case_to_list[4] = []
        case_to_list[5] = []
        case_to_list[6] = []
        case_to_list[7] = []
        for index, row in df.iterrows():
            full_ID = row['Full ID']
            if pd.isnull(full_ID):
                continue
            letter_ID = self.LSM_obj.extract_ending_from_full_id(full_ID)
            if letter_ID != 'G':
                #We need a 'G' collection identifier.
                continue
            ID = row['ID']
            vaccine_dates = row[v_date_cols].count()
            if vaccine_dates < 5:
                #We need at least 5 vaccines to proceed.
                continue
            #At this point we have an individual with a 'G'
            #sample who had at least 5 vaccines.
            sample_doc = row[DOC]
            candidates += 1
            print(f'=====================')
            print(f'{ID=}')
            inf_dates = row[inf_date_cols]
            #Case 1: No infections
            if inf_dates.count() == 0:
                print('Participant had no infections.')
                print('Case 1')
                case_to_list[1].append(index)
                continue

            selector = inf_dates.notnull()
            inf_dates = inf_dates[selector]

            #Case 2: Only infection prior to Dec 15 2022
            dec_15_2021 = datetime.datetime(2021,12,15)
            selection = inf_dates < dec_15_2021

            if selection.all():
                print('Participant had only infections prior to Dec-15-2021')
                print('Case 2')
                case_to_list[2].append(index)
                continue

            jun_30_2022 = datetime.datetime(2022,6,30)
            #Case 3: All infections before or at Jun 30 2022
            #AND
            #Case 3: At least one infection between Dec 15 2021 and Jun 30 2022
            selection  = inf_dates   <= jun_30_2022
            constraint_1 = selection.all()

            selection  &= dec_15_2021 <= inf_dates
            constraint_2 = selection.any()

            if constraint_1 and constraint_2:
                txt = ('Participant had only infections before'
                        ' or on Jun-30-2022 and at least one'
                        ' infection after or on Dec-15-2021')
                print(txt)
                print('Case 3')
                case_to_list[3].append(index)
                continue

            #Case 4/5: At least one infection between 
            #Dec 15 2021 and Jun 30 2022 (inclusive)
            #AND
            #Case 4/5: At least one infection after Jun 30 2022
            selection  = inf_dates   <= jun_30_2022
            selection  &= dec_15_2021 <= inf_dates
            constraint_1 = selection.any()

            selection    = jun_30_2022 < inf_dates
            constraint_2 = selection.any()

            if constraint_1 and constraint_2:
                txt = ('Participant had at least one infection'
                        ' between Dec-15-2021 and Jun-30-2022'
                        ' (inclusive), and at least one infection'
                        ' after Jun-30-2022')
                print(txt)
                ba5_infection_dates = inf_dates[selection]
                first_ba5_infection_date = ba5_infection_dates.iloc[0]
                if sample_doc < first_ba5_infection_date:
                    #Case 4
                    print('>>>Sample was collected before the first BA.5')
                    print('Case 4')
                    case_to_list[4].append(index)
                    continue
                else:
                    #Case 5
                    print('>>>Sample was collected after or at the first BA.5')
                    print('Case 5')
                    case_to_list[5].append(index)
                    continue


            #Case 6/7: Only infections after Jun 30 2022
            selection    = jun_30_2022 < inf_dates
            constraint_1 = selection.all()

            if constraint_1:
                print('Participant had only infections after Jun-30-2022')
                ba5_infection_dates = inf_dates[selection]
                first_ba5_infection_date = ba5_infection_dates.iloc[0]
                if sample_doc < first_ba5_infection_date:
                    #Case 6
                    print('>>>Sample was collected before the first BA.5')
                    print('Case 6')
                    case_to_list[6].append(index)
                    continue
                else:
                    #Case 7
                    print('>>>Sample was collected after or at the first BA.5')
                    print('Case 7')
                    case_to_list[7].append(index)
                    continue

        print(f'We have {candidates=}.')
        fname = 'stratification_inf_vac_nov_30_2022.xlsx'
        folder= 'X'
        fname = os.path.join(self.parent.requests_path,
                folder, fname)
        with pd.ExcelWriter(fname) as writer:
            for key, L in case_to_list.items():
                #The key represents the case
                #L is the list of full_IDs for a given case.
                print(f'For case #{key} we have {len(L)} elements.')
                df_slice = df.loc[L,:].copy()
                sh_name = 'case_' + str(key)
                df_slice.to_excel(writer, sheet_name = sh_name, index = False)

    def map_old_ids_to_new(self, df):
        #Dec 06 2022
        #This function has to be tested.
        fname = 'mapping_old_ids_to_new.xlsx'
        fname = os.path.join(self.parent.MPD_path, fname)
        mapping = pd.read_excel(fname)
        for index_map, row_map in mapping.iterrows():
            old_id = row_map['Old']
            new_id = row_map['New']
            df['ID'].replace(old_id, new_id, inplace=True)
        selection = df['ID'] == 'X'
        if selection.any():
            df.drop(selection[selection].index, inplace=True)



    def generate_infection_and_reason_dict(self, df_up):
        #For infections from the Schlegel Village
        #we use the format month/day/year
        self.use_day_first_for_slash = False
        flag_update_active = False
        flag_update_waves  = False
        infection_dictionary = {'ID':[], 'date':[]}
        reason_dictionary = {'ID':[],
                             self.reason:[],
                             'date':[]}
        #fname = 'up.xlsx'
        #folder = 'Tara_oct_31_2022'
        #fname = os.path.join(self.requests_path, folder, fname)
        #df_up = pd.read_excel(fname, header=None)
        #df_up.dropna(axis=0, inplace=True)

        id_rx     = re.compile('[0-9]{2}[-][0-9]{7}')
        #The following regexp is no longer in use.
        #reason_rx = re.compile('[a-zA-Z]+([ ][a-zA-Z]+)*')
        #Moved Out --> Moved
        #Now we use the following.
        reason_rx = re.compile('[-a-zA-Z]+')

        #Intead of using a REGEXP for the date, we use
        #the functions we created in the MPD Class.
        #date_rx   = re.compile('[a-zA-Z]+[ ]+[0-9]{1,2}(?P<year>[ ]+[0-9]+)?')

        #We assume the data frame has only one column.
        for txt in df_up[0]:
            print(txt)
            #date_obj = date_rx.search(txt)
            date, matched_str = self.convert_str_to_date(txt)
            txt_m_date = txt.replace(matched_str, '')
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

            print(f'{ID=}')
            print(f'{reason=}')
            print(f'{date=}')
            print('---------Extraction is complete.')
            selector = self.parent.df['ID'] == ID
            if reason.lower() in self.removal_states_l:
                #This individual has been removed
                flag_update_active = True
                reason_dictionary['ID'].append(ID)
                reason_dictionary[self.reason].append(reason)
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
        #01-8580579 Deceased 13/08/2022
        #50-1910008 Deceased Sep 15 2022
        #14-5077158  Positive Oct 2 2022
        #>>>Removed 14-5077158  Positive Oct 2 (No year)


        (flag_update_active,
                flag_update_waves,
                infection_dictionary,
                reason_dictionary) =\
                        self.generate_infection_and_reason_dict(df_up)

        if flag_update_active:
            print('Updating MPD Reason or DOR')
            DOR = self.DOR
            df_up = pd.DataFrame(reason_dictionary)
            df_up[DOR] = pd.to_datetime(df_up['date'])
            df_up.drop(columns='date', inplace=True)
            #Replace old IDs with new
            self.map_old_ids_to_new(df_up)
            print(df_up)
            #self.update_reason_dates_and_status(df_up)
            #Merge instead of point modifications
            self.parent.df = self.parent.merge_with_M_and_return_M(df_up,
                    'ID', kind='update+')
            self.update_active_status_column()
        if flag_update_waves:
            df_up = pd.DataFrame(infection_dictionary)
            df_up['DOI'] = pd.to_datetime(df_up['date'])
            #Replace old IDs with new
            self.map_old_ids_to_new(df_up)
            print(df_up)
            self.parent.LIS_obj.update_the_dates_and_waves(df_up)
            #Is this necessary?
            self.update_active_status_column()
        print('Please write to Excel externally.')

    def load_single_column_df_for_update(self, fname, folder, sheet=0):
        #Dec 05 2022
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df_up = pd.read_excel(fname, header=None, sheet_name=sheet)
        df_up.dropna(axis=0, inplace=True)
        return df_up

    def single_column_update(self):
        #Use this function for updates using 
        #the one-column format.
        fname  = 'updates.xlsx'
        folder = 'Tara_mar_21_2023'
        df_up = self.load_single_column_df_for_update(fname, folder)
        print(df_up)
        #Pre-update
        status_pre = self.compute_data_density(self.parent.df)
        #Recalculate the age of all participants.
        self.compute_age_from_dob(self.parent.df)
        #Intermediate step
        #=======================================================
        #This is reserved for special requests
        #=======================================================
        #for index, row in self.parent.df.iterrows():
            #ID = row['ID']
            #site = ID[:2]
            #sep_30 = datetime.datetime(2022,9,30)
            #sep_17 = datetime.datetime(2022,9,17)
            #if site == '03':
                #if pd.notnull(row[self.reason]):
                    ##Do not modify if the reason is available
                    #print(f'{row[self.reason]=}')
                    #continue
                #self.parent.df.loc[index, self.DOR] = sep_30
                #self.parent.df.loc[index, self.reason] = self.RC
            #if site == '20' or site == '61':
                #if pd.notnull(row[self.reason]):
                    ##Do not modify if the reason is available
                    #print(f'{row[self.reason]=}')
                    #continue
                #self.parent.df.loc[index, self.DOR] = sep_17
                #self.parent.df.loc[index, self.reason] = self.RC

        #Update step
        self.extract_and_update_DOR_Reason_Infection(df_up)
        #Post-update
        status_post = self.compute_data_density(self.parent.df)
        self.monotonic_increment_check(status_pre,
                status_post)

    def two_column_update(self):
        #Use this function for updates using 
        #the one-column format.
        fname  = 'update_2.xlsx'
        folder = 'Megan_dec_05_2022'
        df_up = self.load_single_column_df_for_update(fname, folder)
        df_up[0] = df_up[0].str.replace('LTC1-','')
        txt = 'Refused Extension - Withdrawn on'
        df_up[1] = df_up[1].str.replace(txt,'Refused-Consent')
        txt = 'No Reconsent - Withdraw'
        df_up[1] = df_up[1].str.replace(txt,'Refused-Consent')
        df_up[1] = df_up[1].str.replace(',','')
        df_up[0] = df_up[0] + ' ' + df_up[1]
        print(df_up)
        self.extract_and_update_DOR_Reason_Infection(df_up)


    #Nov 25 2022
    def whole_blood_update(self):
        folder = 'Megan_nov_25_2022'
        fname = 'whole_blood.xlsx'
        fname = os.path.join(self.parent.requests_path,folder, fname)
        df_up = pd.read_excel(fname)
        print(df_up)
        self.parent.df = self.merge_X_with_Y_and_return_Z(self.parent.df,
                                         df_up,
                                         self.merge_column,
                                         kind='original+')

    def compute_data_density(self, df):
        #Dec 23 2022
        #Compute how many cells are occupied.
        #This function is also called within the LSM class.
        col_to_n_data = {'Column':[], 'Count':[], 'Empty':[], '%_full':[]}
        n_rows = len(df)
        for column in df.columns:
            n_data = df[column].count()
            col_to_n_data['Column'].append(column)
            col_to_n_data['Count'].append(n_data)
            col_to_n_data['Empty'].append(n_rows-n_data)
            p = n_data/n_rows*100
            col_to_n_data['%_full'].append(p)
        df_from_dict = pd.DataFrame(col_to_n_data)
        return (df_from_dict, n_rows)


    def monotonic_increment_check(self, pre, post):
        #Dec 23 2022
        #Create a delta report between the pre- and
        #post-merging events.
        #The delta_report is stored in this class.
        df_pre  = pre[0]
        df_post = post[0]
        n_rows_pre  = pre[1]
        n_rows_post = post[1]
        M = pd.merge(df_pre, df_post,
                on='Column',
                how='outer',
                suffixes=('_old', '_current'))
        pre_total = df_pre['Count'].sum()
        post_total = df_post['Count'].sum()
        p_increment = (post_total - pre_total)/pre_total * 100
        M['Delta Count']  = M['Count_current'] - M['Count_old']
        M['Delta %_full'] = M['%_full_current'] - M['%_full_old']
        check = M['Delta Count'] >= 0
        if not check.all():
            raise ValueError('There seems to be some data loss')
        L = ['Count_old','Count_current']
        M.loc['Total'] = M[L].sum(axis=0)
        print(M)
        print(f'Total pre--data:{pre_total}')
        print(f'Total post-data:{post_total}')
        print(f'% increment    :{p_increment}')
        self.delta_report = M

    def peace_of_mind_check(self):
        fname = 'column_types.xlsx'
        fname = os.path.join(self.parent.outputs_path, fname)
        df_t = pd.read_excel(fname)
        #Columns that should not have spaces
        print('>>>Making sure there are no unexpected spaces')
        for index_t, row_t in df_t.iterrows():
            col_name = row_t['Name']
            spaces_allowed = row_t['Spaces allowed?']
            if spaces_allowed == 'No':
                O = self.parent.df[col_name]
                T = self.parent.df[col_name].str.replace(' ','')
                if O.equals(T):
                    pass
                else:
                    print('There were spaces within', col_name)
                    raise ValueError('Spaces are not allowed')
                self.parent.df[col_name] = T
        print('There are no unexpected spaces.')

        self.check_for_repeats()

        self.parent.check_id_format(self.parent.df, self.merge_column)

        sex_allowed_values = ['Male', 'Female', np.nan]
        S = self.parent.df['Sex'].isin(sex_allowed_values)
        if not S.all():
            print(self.parent.df.loc[~S,header])
            raise ValueError('Sex column is not compliant.')
        else:
            print('Sex column is compliant.')

        #self.removal_states = ['Deceased', 'Discharged', 'Moved',
                               #'Declined', 'Withdrew', self.RC]
        reason_allowed_values = self.removal_states.copy()
        reason_allowed_values += ['Refused-Combative', 'Refused-Palliative']
        reason_allowed_values += ['Refused-NoContact', 'Refused-HealthReasons']
        reason_allowed_values += ['Invalid-Consent', np.nan]
        S = self.parent.df['Reason'].isin(reason_allowed_values)
        if not S.all():
            print(self.parent.df.loc[~S,header])
            raise ValueError('Reason column is not compliant.')
        else:
            print('Reason column is compliant.')



        vacc_allowed_values = self.parent.LIS_obj.list_of_valid_vaccines
        vacc_allowed_values += [np.nan]
        h = 'Vaccine Type '
        for k in range(1,5+1):
            header = h + str(k)
            S = self.parent.df[header].isin(vacc_allowed_values)
            if not S.all():
                print(header)
                print(self.parent.df.loc[~S,header])
                raise ValueError('Vaccine column is not compliant.')
            else:
                print(header, 'column is compliant.')

        inf_allowed_values = ['PCR','RAT']
        inf_allowed_values += [np.nan]
        h = 'Infection Type '
        for k in range(1,6+1):
            header = h + str(k)
            S = self.parent.df[header].isin(inf_allowed_values)
            if not S.all():
                print(header)
                print(self.parent.df.loc[~S,header])
                raise ValueError('Infection column is not compliant.')
            else:
                print(header, 'column is compliant.')

