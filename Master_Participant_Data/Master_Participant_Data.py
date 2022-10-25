#JRR @ McMaster University
#Update: 18-Sep-2022
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

        self.parent = None

        self.merge_column = 'ID'
        self.DOR      = 'Date Removed from Study'
        self.DOE      = 'Enrollment Date'
        self.note     = 'Notes/Comments'
        self.reason   = 'Reason'
        self.status   = 'Active'
        self.comments = 'Notes/Comments'
        self.removal_states = ['Deceased', 'Discharged', 'Moved Out',
                               'Decline', 'Withdraw from Study']
        self.removal_states_l = [x.lower() for x in self.removal_states]

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


    def relabel_ids(self):
        #We aim for a consistent and simple naming of variables 
        dc = {'Sample ID':'ID', 'Enrollment Date (dd-mm-yyyy)':'Enrollment Date'}
        self.parent.df.rename(columns=dc, inplace=True)


    def delete_unnecessary_columns(self):
        self.parent.df.drop(columns=['Inventory File', 'Combo'], inplace=True)


    def remove_nan_ids(self):
        self.parent.df.dropna(axis=0, subset=['ID'], inplace=True)

    def check_for_repeats(self):
        #We drop rows containing NA in the "Sample ID" from the
        #file because sometimes there are additional comments
        #below the table that get interpreted as additional rows.
        self.parent.df.dropna(subset=[self.merge_column], inplace=True)
        value_count_gt_1 = self.parent.df[self.merge_column].value_counts().gt(1)
        if value_count_gt_1.any():
            print('We have repetitions in Master Participant Data')
            print('Check the column:', self.merge_column)
            raise Exception('No repetitions were expected.')
        else:
            print('No repeats were found in Master Participant Data')

    def compute_DOB_from_enrollment(self):
        #Estimate the DOB in case it is missing
        dob = 'DOB'
        #Convert integer age baseline to days.
        age_in_days = pd.to_timedelta(self.parent.df['Age Baseline'] * 365, unit='D')
        doe= 'Enrollment Date'
        no_consent = 'no consent received'
        exceptions = [no_consent, '00', '#N/A']
        selection = self.parent.df[doe].isin(exceptions)
        if selection.any():
            #Note that we are using selection as a boolean
            #vector for the loc argument of the selection 
            #object itself.
            indices = selection.loc[selection]
            print('There are dates in the set of exceptions.')
            #Print exceptions
            for index, _ in indices.iteritems():
                individual = self.parent.df.loc[index, 'ID']
                exception = self.parent.df.loc[index, doe]
                print('The date of enrollment for:', individual, end='')
                print(' is:', exception)
            #Relabel exception       
            self.parent.df.loc[selection,doe] = np.nan
        #Format as date
        self.parent.df[doe] = pd.to_datetime(self.parent.df[doe], dayfirst=True)
        #Compute DOB as DOE minus Age in days if DB does not exists.
        T = self.parent.df[doe] - age_in_days
        #Replace only if empty
        self.parent.df[dob] = self.parent.df[dob].where(~self.parent.df[dob].isnull(), T)


    def clean_date_removed_from_study(self):
        col_name = 'Date Removed from Study'
        L = []
        for index, value in self.parent.df[col_name].iteritems():
            if isinstance(value, str):
                txt = value.lower()
                #Remove spaces
                if value.isspace():
                    L.append(np.nan)
                else:
                    print(txt)
                    L.append(txt)
            else:
                L.append(value)
        #Convert to datetime format
        self.parent.df[col_name] = pd.to_datetime(L)


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


    def update_deceased_discharged(self):
        fname = 'deceased_discharged_16_Sep_2022.xlsx'
        txt = os.path.join(self.dpath, fname)
        df_ids = pd.read_excel(txt)
        up_col = 'Reason'
        for _, update_row in df_ids.iterrows():
            ID = update_row['ID']
            state = update_row[up_col]
            rows = self.parent.df[self.parent.df['ID']== ID]
            for index, row in rows.iterrows():
                value = row[up_col]
                if pd.isnull(value):
                    self.parent.df.loc[index, up_col] = state
                    print('Made a change in the Reason column for', ID)
                    print('Changed to', state)
                else:
                    print('Already had a value for', ID, ':', value)

    ##########Sep 22 2022##################

    def initialize_class_with_df(self, df):
        self.parent.df = df

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


    def generate_excel_file(self):
        fname = 'Master_Participant_Data_X.xlsx'
        txt = os.path.join(self.dpath, fname)
        self.parent.df.to_excel(txt, index=False)
        print('Excel file was produced.')

    def compare_data_frames(self):
        df1 = pd.read_excel('./Master_Participant_Data_Y.xlsx')
        df2 = pd.read_excel('./Master_Participant_Data_X.xlsx')
        if df1.equals(df2):
            print('They are equal')

    def load_main_frame(self):
        fname = 'Master_Participant_Data_X.xlsx'
        fname = os.path.join(self.dpath, fname)
        self.parent.df = pd.read_excel(fname)
        print('Excel file was loaded.')

    def update_active_status_column(self):
        c1 = self.parent.df[self.DOR].notnull()
        c2 = self.parent.df[self.reason].notnull()
        self.parent.df[self.status] = ~(c1 | c2)
        print('The active status column has been updated.')

    def full_run(self):
        self.relabel_ids()
        self.delete_unnecessary_columns()
        self.remove_nan_ids()
        self.check_for_repeats()
        self.compute_DOB_from_enrollment()
        self.clean_date_removed_from_study()
        self.add_Y_in_the_Whole_Blood_column()
        self.update_deceased_discharged()
        self.generate_excel_file()
        #self.compare_data_frames()

        print('Module: module_Master_Participant_Data.py FINISHED')

    ##########Oct 17 2022##################
    def add_site_column(self):
        rexp = re.compile('(?P<site>[0-9]+)[-](?P<user>[0-9]+)')
        def get_site(txt):
            obj = rexp.search(txt)
            if obj:
                return obj.group('site')
            else:
                raise ValueError('Unable to extract site')

        site = 'Site'
        if site in self.parent.df.columns:
            return
        self.parent.df[site] = self.parent.df['ID']
        self.parent.df[site] = self.parent.df[site].apply(get_site)

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


    ##########Oct 25 2022##################
    def compute_age_from_dob(self):
        for index, row in self.parent.df.iterrows():
            dob = row['DOB']
            if pd.isnull(dob):
                continue
            today = datetime.datetime.now()
            delta = (today - dob).days
            years = delta // 365
            self.parent.df.loc[index,'Age'] = years









