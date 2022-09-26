#JRR @ McMaster University
#Update: 18-Sep-2022
import numpy as np
import pandas as pd
import os
import re
import matplotlib as mpl
mpl.rcParams['figure.dpi']=300
import matplotlib.pyplot as plt


# <b> Section: Master_Participant_Data </b>

class MasterParticipantData:
    def __init__(self, path):
        #Which row contains the first data entry in the Excel file
        self.excel_starts_at = 3
        self.dpath = path
        MPD = 'Master_Participant_Data'
        self.fname = os.path.join(self.dpath, MPD + '.xlsx')
        #Read the Excel file containing the data
        self.df = pd.read_excel(self.fname,
                                sheet_name="Master", skiprows=[0])
        self.merge_column = 'ID'
        self.DOR      = 'Date Removed from Study'
        self.reason   = 'Reason'
        self.status   = 'Active'
        self.removal_states = ['deceased', 'moved out',
                               'decline', 'withdraw from study']


    def relabel_ids(self):
        #We aim for a consistent and simple naming of variables 
        dc = {'Sample ID':'ID', 'Enrollment Date (dd-mm-yyyy)':'Enrollment Date'}
        self.df.rename(columns=dc, inplace=True)


    def delete_unnecessary_columns(self):
        self.df.drop(columns=['Inventory File', 'Combo'], inplace=True)


    def remove_nan_ids(self):
        self.df.dropna(axis=0, subset=['ID'], inplace=True)

    def check_for_repeats(self):
        #We drop rows containing NA in the "Sample ID" from the
        #file because sometimes there are additional comments
        #below the table that get interpreted as additional rows.
        self.df.dropna(subset=[self.merge_column], inplace=True)
        value_count_gt_1 = self.df[self.merge_column].value_counts().gt(1)
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
        age_in_days = pd.to_timedelta(self.df['Age Baseline'] * 365, unit='D')
        doe= 'Enrollment Date'
        no_consent = 'no consent received'
        exceptions = [no_consent, '00', '#N/A']
        selection = self.df[doe].isin(exceptions)
        if selection.any():
            #Note that we are using selection as a boolean
            #vector for the loc argument of the selection 
            #object itself.
            indices = selection.loc[selection]
            print('There are dates in the set of exceptions.')
            #Print exceptions
            for index, _ in indices.iteritems():
                individual = self.df.loc[index, 'ID']
                exception = self.df.loc[index, doe]
                print('The date of enrollment for:', individual, end='')
                print(' is:', exception)
            #Relabel exception       
            self.df.loc[selection,doe] = np.nan
        #Format as date
        self.df[doe] = pd.to_datetime(self.df[doe], dayfirst=True)
        #Compute DOB as DOE minus Age in days if DB does not exists.
        T = self.df[doe] - age_in_days
        #Replace only if empty
        self.df[dob] = self.df[dob].where(~self.df[dob].isnull(), T)



    def clean_date_removed_from_study(self):
        col_name = 'Date Removed from Study'
        L = []
        for index, value in self.df[col_name].iteritems():
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
        self.df[col_name] = pd.to_datetime(L)


    def get_vaccine_types_and_dates(self):
        vaccine_type_cols = []
        vaccine_dates_cols = []
        vt = 'vaccine type'
        vd = 'vaccine date'
        for column in self.df.columns:
            c = column.lower()
            if vt in c:
                vaccine_type_cols.append(column)
            elif vd in c:
                vaccine_dates_cols.append(column)
        tpl = (vaccine_type_cols, vaccine_dates_cols)
        return tpl


    def print_col_names_and_types(self):
        for name, dtype in zip(self.df.columns, self.df.dtypes):
            print(name, ':', dtype)


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
            rows = self.df[self.df['ID']== ID]
            for index, row in rows.iterrows():
                if row[up_col] != 'Y':
                    self.df.loc[index, up_col] = state
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
            rows = self.df[self.df['ID']== ID]
            for index, row in rows.iterrows():
                value = row[up_col]
                if pd.isnull(value):
                    self.df.loc[index, up_col] = state
                    print('Made a change in the Reason column for', ID)
                    print('Changed to', state)
                else:
                    print('Already had a value for', ID, ':', value)

    ##########Sep 22 2022##################

    def initialize_class_with_df(self, df):
        self.df = df

    def update_reason_dates_and_status(self, df_up):
        print('Update the Reason, the DOR and the status (Active).')
        for index, row in df_up.iterrows():
            ID = row['ID']
            date = row['date']
            reason = row[self.reason]
            selector = self.df['ID'] == ID
            self.df.loc[selector, self.DOR] = date
            self.df.loc[selector, self.reason] = reason

        print('The DOR and Reason columns have been updated.')

        self.update_active_status_column()


    def generate_excel_file(self):
        fname = 'Master_Participant_Data_X.xlsx'
        txt = os.path.join(self.dpath, fname)
        self.df.to_excel(txt, index=False)
        print('Excel file was produced.')

    def compare_data_frames(self):
        df1 = pd.read_excel('./Master_Participant_Data_Y.xlsx')
        df2 = pd.read_excel('./Master_Participant_Data_X.xlsx')
        if df1.equals(df2):
            print('They are equal')

    def load_main_frame(self):
        fname = 'Master_Participant_Data_X.xlsx'
        fname = os.path.join(self.dpath, fname)
        self.df = pd.read_excel(fname)
        print('Excel file was loaded.')

    def update_active_status_column(self):
        c1 = self.df[self.DOR].notnull()
        c2 = self.df[self.reason].notnull()
        self.df[self.status] = ~(c1 | c2)
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



