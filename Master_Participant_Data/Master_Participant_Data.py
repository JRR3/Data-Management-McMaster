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

        self.parent = None

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
        self.removal_states = ['Deceased', 'Discharged', 'Moved',
                               'Declined', 'Withdrew', 'Refused-Consent']
        self.removal_states_l = [x.lower() for x in self.removal_states]

        self.yearfirst_regexp = re.compile('[0-9]{4}[-][0-9]{2}[-][0-9]{2}')

        self.dayfirst_regexp = re.compile('[0-9]+[-][a-zA-Z]+[-](?P<year>[0-9]+)')

        self.dayfirst_with_slash_regexp = re.compile('[0-9]+[/][0-9]+[/](?P<year>[0-9]+)')

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
        self.parent.df.dropna(subset=[self.merge_column], inplace=True)
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


    ##########Oct 17 2022##################
    def add_site_column(self):
        rexp = re.compile('(?P<site>[0-9]+)[-](?P<user>[0-9]+)')
        def get_site(txt):
            obj = rexp.search(txt)
            if obj:
                return int(obj.group('site'))
            else:
                raise ValueError('Unable to extract site')

        site = 'Site'
        if site in self.parent.df.columns:
            return
        self.parent.df[site] = self.parent.df['ID']
        self.parent.df[site] = self.parent.df[site].apply(get_site)
        self.parent.df[self.site_type] = 'LTC'
        self.parent.df[self.site_type] =\
                self.parent.df[self.site_type].where(
                        self.parent.df[site] < 50, 'RH')

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


    def convert_str_to_date(self, txt, use_day_first_for_slash=True):
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
                    date = obj.group(0)
                    date = self.short_year_to_long_year(date)
                    date = pd.to_datetime(date, dayfirst=True)
                else:
                    print(f'{txt=}')
                    raise ValueError('Unknown format for date.')
        return (date, obj.group(0))









