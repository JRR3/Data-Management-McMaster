#JRR @ McMaster University
#Update: 18-Sep-2022
import numpy as np
import pandas as pd
import os
import re
import datetime
import shutil
import matplotlib as mpl
mpl.rcParams['figure.dpi']=300
import matplotlib.pyplot as plt
from tqdm import tqdm as pbar


# <b> Section: Master_Participant_Data </b>

class LTCSerologyMaster:
    def __init__(self, dpath, parent=None):
        self.parent       = None
        self.df           = None
        self.serology_codes = None
        self.dpath        = dpath
        self.backups_path = os.path.join('..', 'backups')
        self.merge_source = 'Full ID'
        self.merge_column = 'ID'
        self.ID_columns   = [self.merge_source,
                self.merge_column]
        self.DOC          = 'Date Collected'
        self.non_numeric_columns = []
        self.numeric_columns     = []
        self.ancode_to_lcode     = {}
        self.lcode_to_ancode     = {}

        self.load_LSM_file()


        if parent:
            self.parent = parent
            print('LSM class initialization from Manager.')
        else:
            raise ValueError('Parent object is unavailable.')

    def load_LSM_file(self):
        #Note that the LSM file already has compact format, i.e.,
        #the headers are simple.

        #Read the Excel file containing the data
        fname = 'LSM.xlsx'
        fname = os.path.join(self.dpath, fname)
        self.df = pd.read_excel(fname)

        #Read the Excel file containing the 'lookup_table_S_codes'
        fname = 'lookup_table_S_codes.xlsx'
        fname = os.path.join(self.dpath, fname)
        self.serology_codes = pd.read_excel(fname)

        self.non_numeric_columns = self.ID_columns + [self.DOC]
        for column in self.df.columns:
            if column not in self.non_numeric_columns:
                self.numeric_columns.append(column)

        print('LSM class has been initialized with the LSM file.')

        self.generate_letter_to_AN_code_dict()


    def update_id_column(self):
        #This function was updated on Oct 31, 2022
        self.df = self.parent.create_df_with_ID_from_full_ID(self.df)

    ######September 28 20222

    def backup_the_LSM_file(self):
        fname = 'LSM.xlsx'
        original = os.path.join(self.dpath, fname)
        today = datetime.datetime.now()
        date  = today.strftime('%d_%m_%Y_time_%H_%M_%S')
        bname = 'LSM' + '_backup_' + date
        bname += '.xlsx'
        backup   = os.path.join(self.backups_path, bname)
        shutil.copyfile(original, backup)
        print('A backup for the LSM file has been generated.')

    def write_LSM_to_excel(self):
        #Dec 23 2022
        #Note that we are using the delta report
        #stored in the MPD class.
        self.backup_the_LSM_file()

        print('Writing the LSM file to Excel.')

        fname = 'LSM.xlsx'
        fname = os.path.join(self.dpath, fname)
        with pd.ExcelWriter(fname) as writer:
            self.df.to_excel(writer,
                    sheet_name = 'data', index = False)
            #########
            if self.parent.MPD_obj.delta_report is None:
                pass
            else:
                print('Writing the Delta report to Excel')
                self.parent.MPD_obj.delta_report.to_excel(writer,
                        sheet_name = 'report', index = False)
            #########

        print('The LSM file has been written to Excel.')


    def map_old_ids_to_new(self, df_up):
        #Careful with the "E" type individuals.
        fname  = 'remapping_list.xlsx'
        fname = os.path.join(self.dpath, fname)
        df_re = pd.read_excel(fname)
        #This participant changed sites.
        df_up[self.merge_source] = df_up[self.merge_source].str.replace('50-1910060',
                '12-1301348')
        for index_re, row_re in df_re.iterrows():
            old_id = row_re['Original']
            new_id = row_re['New']
            df_up[self.merge_source].replace(old_id, new_id, inplace=True)

    def update_LND_data(self):
        #Updated on Nov 22, 2022
        fname  = 'LND_update.xlsx'
        folder = 'Jessica_nov_22_2022'
        fname = os.path.join('..','requests',folder, fname)
        df_up = pd.read_excel(fname)
        #If no Full ID is provided, the row is removed.
        df_up.dropna(axis=0, subset=self.merge_source, inplace=True)
        #Remap faulty "Full IDs"
        self.map_old_ids_to_new(df_up)
        kind = 'original+'
        merge_at_column = 'Full ID'
        self.df = self.parent.merge_X_with_Y_and_return_Z(self.df,
                                                          df_up,
                                                          merge_at_column,
                                                          kind=kind)
        self.update_id_column()
        print('End of LSM updating process.')

    def what_is_missing(self):
        col_to_ids = {}
        max_count = -1
        for column in self.df.columns:
            if column in self.non_numeric_columns:
                continue
            selection = self.df[column].isnull()
            n_of_empty_cells = selection.value_counts()[True]
            if max_count < n_of_empty_cells:
                max_count = n_of_empty_cells
            df_s = self.df[selection]
            IDs  = df_s[self.merge_source].values
            col_to_ids[column] = IDs
        print(f'{max_count=}')
        for key, L in col_to_ids.items():
            n_of_empty_cells = len(L)
            n_nans = max_count - n_of_empty_cells
            col_to_ids[key] = np.pad(L, (0,n_nans), 'empty')
        df_m = pd.DataFrame(col_to_ids)
        label = 'missing_data_for_serology_samples'
        self.parent.write_df_to_excel(df_m, label)

    def extract_ending_from_full_id(self, full_id):
        #Dec 23 2022
        txt = '[0-9]{2}[-][0-9]{7}[-](?P<collection>[0-9a-zA-Z]+)'
        regexp = re.compile(txt)
        obj = regexp.match(full_id)
        if obj:
            return obj.group('collection')
        else:
            raise ValueError('Unable to extract collection identifier.')

    def serology_decay_computation(self):
        #Tara requested these computations
        #on Nov 29 2022
        #Last update: Nov 30 2022
        print('Warning: Make sure the W file is updated.')
        fname  = 'W.xlsx'
        fname = os.path.join(self.parent.outputs_path, fname)
        df_w = pd.read_excel(fname)
        DOC = self.DOC
        marker = 'Wuhan (SB3)'

        n_levels = 4
        levels = list(range(1,n_levels+1))
        level_to_list = {}
        level_to_n_cases_wo_posterior = {}
        for level in levels:
            level_to_list[level] = []
            level_to_n_cases_wo_posterior[level] = 0

        level_to_delta_vac = {1:40, 2:8*30, 3:6*30, 4:9*30}
        #Careful with the last case.
        level_to_n_vacs = {1:2, 2:3, 3:4, 4:5}

        #List of tuples (x,y,z)
        #x=Full ID
        #y=DOC (date of collection)
        #z=Wuhan MNT value

        v_date_cols = self.parent.LIS_obj.vaccine_date_cols
        inf_date_cols = self.parent.LIS_obj.positive_date_cols
        #Iterate over the M file
        for index_m, row_m in self.parent.df.iterrows():
            ID = row_m['ID']
            vaccine_dates = row_m[v_date_cols]
            for level in levels:
                #For a given level we have a constraint
                #for the distance between
                #vaccine_dates[level-1] and
                #vaccine_dates[level].
                #Moreover, the sample has to
                #be taken between
                #vaccine_dates[level] and
                #vaccine_dates[level+1] (if it exists).

                n_vacs_req = level_to_n_vacs[level]
                n_vacs    = vaccine_dates.count()

                if n_vacs < n_vacs_req:
                    #We need at least "n_vac_req"
                    #vaccines to proceed.

                    #No more levels to check.
                    #Move to next participant.
                    break

                #At this point we have an individual with at
                #least n_vac_req
                #Note: Level 1 = 2nd dose
                #Note: Level 2 = 3rd dose
                #Note: Level 3 = 4th dose
                #Note: Level 4 = 5th dose
                previous_dose   = vaccine_dates[level-1]
                current_dose    = vaccine_dates[level]
                posterior_dose  = datetime.datetime(2023,1,1)
                v_plus          = n_vacs_req + 1
                no_posterior_dose = False
                if v_plus <= n_vacs:
                    #This person has at least one more vaccination
                    posterior_dose  = vaccine_dates[level+1]
                else:
                    print(f'{ID=} does not have the additional dose {v_plus}.')
                    print(f'However, we can still proceed.')
                    no_posterior_dose = True
                    #if level == 4:
                        #pass
                    #else:
                        ##Force the requirement of the posterior dose.
                        #break
                delta  = current_dose - previous_dose
                delta /= np.timedelta64(1,'D')
                if delta < 0:
                    raise ValueError('Time delta cannot be negative.')
                max_delta = level_to_delta_vac[level]
                if max_delta < delta:
                    #We need less than or equal to
                    #delta days between the current
                    #and previous vaccinations.

                    #No more levels to check.
                    #Move to next participant.
                    break
                #Get samples for this individual
                selection = self.df['ID'] == ID
                if not selection.any():
                    print(f'{ID=} has no samples.')
                    #No more levels to check.
                    #Move to next participant.
                    break
                #Nonempty LSM samples
                samples = self.df[selection]
                #Iterate over samples
                for index_s, row_s in samples.iterrows():
                    full_ID = row_s['Full ID']
                    wuhan_s = row_s[marker]
                    if pd.isnull(wuhan_s):
                        print(f'Sample {full_ID=} has no Wuhan.')
                        continue
                    #The sample had to be collected between the
                    #current and posterior doses. 
                    #In other words,
                    #the sample was collected
                    #between the (level)
                    #and (level+1) vaccines 
                    #(if it exists)
                    doc_s = row_s[DOC]
                    if current_dose <= doc_s and doc_s <= posterior_dose:
                        print(f'{full_ID=} is between the current and posterior dose.')
                        print(f'{full_ID=} is between dose {level+1} and dose {level+2}.')
                    else:
                        #Move to next sample
                        continue
                    #Now is time to check if no infections
                    #took place before or at the sample collection
                    infection_dates = row_m[inf_date_cols]
                    if infection_dates.count() == 0:
                        print(f'{ID=} never had infections.')
                        #t = (full_ID, doc_s, wuhan_s)
                        #L.append(t)
                        level_to_list[level].append(full_ID)

                        if no_posterior_dose:
                            level_to_n_cases_wo_posterior[level] += 1

                        #Move to next sample.
                        continue
                    selection = infection_dates.notnull()
                    infection_dates = infection_dates[selection]
                    constraint = doc_s < infection_dates
                    if constraint.all():
                        #No infection at or before
                        #the date of sample collection.
                        print(f'{ID=} had no infections before or'
                                ' at the time of sample collection.')
                        #t = (full_ID, doc_s, wuhan_s)
                        #L.append(t)
                        level_to_list[level].append(full_ID)

                        if no_posterior_dose:
                            level_to_n_cases_wo_posterior[level] += 1

                        #Move to next sample.
                        continue


        fname = 'decay_data.xlsx'
        folder= 'Tara_nov_30_2022'
        fname = os.path.join(self.parent.requests_path,
                folder, fname)

        with pd.ExcelWriter(fname) as writer:
            print('Here')
            for key,L in level_to_list.items():
                print(f'For level #{key} we have {len(L)} elements.')
                x = level_to_n_cases_wo_posterior[key]
                cases_with_no_posterior_dose = x
                print(f'{cases_with_no_posterior_dose=}.')
                if len(L) == 0:
                    continue
                selection = df_w['Full ID'].isin(L)
                df_slice = df_w.loc[selection,:].copy()
                min_date = df_slice[DOC].min()
                print(f'For level #{key} {min_date=}.')
                days_col = (df_slice[DOC] - min_date) / np.timedelta64(1,'D')
                df_slice.insert(3,'Days since earliest sample', days_col)
                df_slice.insert(4,'Log2-Wuhan', np.log2(df_slice[marker]))
                sh_name = 'dose_' + str(key+1)
                df_slice.to_excel(writer, sheet_name = sh_name, index = False)


    def plot_decay_for_serology(self):
        marker     = 'Wuhan (SB3)'
        log_marker = 'Log2-Wuhan'
        dses       = 'Days since earliest sample'

        fname = 'decay_data.xlsx'
        folder= 'Tara_nov_30_2022'
        fname = os.path.join(self.parent.requests_path,
                folder, fname)
        book = pd.read_excel(fname, sheet_name=None)
        for k, (sheet, df) in enumerate(book.items()):
            print(f'Working with {sheet=}')
            duration  = df[dses].max()
            min_date  = df[self.DOC].min()
            max_date  = df[self.DOC].max()
            min_day   = min_date.day
            max_day   = max_date.day
            lower_label = min_date.replace(day=1)
            upper_label = max_date.replace(day=1)
            lower_range = lower_label - pd.DateOffset(days=1)
            upper_range = upper_label + pd.DateOffset(months=1)
            print(f'{max_date=}')
            print(f'{min_date=}')
            print(f'{lower_label=}')
            print(f'{upper_label=}')
            print(f'{lower_range=}')
            print(f'{upper_range=}')
            intervals = pd.date_range(start=lower_range,
                    end=upper_range, freq='M')
            periods   = pd.period_range(start=lower_label,
                    end=upper_label, freq='M')
            periods   = periods.to_timestamp().strftime("%b-%y")
            bins = pd.cut(df[self.DOC], intervals, labels=periods)
            df_g = df.groupby(bins)
            plt.close('all')
            ax = df_g.boxplot(subplots=False,
                    #positions=range(len(periods)),
                    rot = 45,
                    column=log_marker,
                    #showfliers=False,
                    )
            reg_exp= re.compile('[0-9]+')
            obj = reg_exp.search(sheet) 
            dose_n = obj.group(0)
            txt = 'Dose ' + dose_n
            ax.set_title(txt)
            #ax.set_xlabel('')
            ax.set_ylabel('$\log_2$(MNT50) (Wuhan)')

            fname = 'dose_' + dose_n + '_box_plot.png'
            folder = 'Tara_nov_30_2022'
            fname = os.path.join(self.parent.requests_path, folder, fname)
            n_cols = len(periods)
            plt.xticks(range(1,n_cols+1),periods)
            ax.figure.savefig(fname)

            #Regression
            log_wuhan_medians = df_g[log_marker].agg('median').dropna()
            days_medians = df_g[dses].agg('median').dropna()
            print(f'{log_wuhan_medians=}')
            print(f'{days_medians=}')
            plt.close('all')
            ax = df.plot.scatter(x=dses, y=log_marker, c='Blue')
            p = np.polyfit(df[dses], df[log_marker], 1)
            #p = np.polyfit(days_medians,
                    #log_wuhan_medians, 1)
            t = np.linspace(0,duration,300)
            y = np.polyval(p, t)
            ax.plot(t,y,'k-',linewidth=2)
            half_life = -1/p[0]
            txt  = 'Dose ' + dose_n
            txt += ': $\lambda={0:.1f}$ days'.format(half_life)
            print(txt)
            ax.set_title(txt)
            ax.set_ylabel('$\log_2$(MNT50) (Wuhan)')
            fname = 'dose_' + dose_n + '_scatter_plot.png'
            folder = 'Tara_nov_30_2022'
            fname = os.path.join(self.parent.requests_path, folder, fname)
            ax.figure.savefig(fname)



    def direct_serology_update_with_headers(self, df_up=None):
        #Dec 22 2022
        #This is now the official method to update the
        #serology master file.
        #Note that this function is called from the
        #update_LSM method within the parent object.
        if df_up is None:
            fname  = 'NazyAbData_2022_11_14.xlsx'
            print(f'Working with {fname=}')
            folder = 'Jessica_dec_07_2022'
            fname = os.path.join(self.parent.requests_path,
                    folder, fname)
            df_up = pd.read_excel(fname, sheet_name='Sept 28 2022')
        print(f'The update has {len(df_up)} rows.')
        #Replace old IDs with the new.
        df_up.dropna(axis=0, how='all', inplace=True)
        df_up.replace('.', np.nan, inplace=True)
        df_up.replace('n/a', np.nan, inplace=True)
        df_up.replace('N/A', np.nan, inplace=True)
        df_up.replace('#N/A', np.nan, inplace=True)
        df_up.replace('retest on next plate', np.nan, inplace=True)
        df_up.replace('NT', np.nan, inplace=True)
        df_up['p Full ID'] = df_up['p Full ID'].str.replace(' ','')
        df_up['p Full ID'] = df_up['p Full ID'].str.replace('wks','wk')
        #=========== Removal of cutoff===============
        selection = df_up['p Full ID'].str.lower().str.contains('cutoff')
        if selection.any():
            df_up = df_up[~selection].copy()
        #===========ID verification===============
        #rexp_c = re.compile('[0-9]{2}[-][0-9]{7}[-][A-Z]{1,2}')
        #def is_a_valid_id(txt):
            #obj = rexp_c.search(txt)
            #if obj:
                #return obj.group(0)
            #else:
                #print(txt)
                #raise ValueError('Not an ID')
        #df_up[self.merge_source] = df_up[self.merge_source].map(is_a_valid_id)
        self.check_full_id_format(df_up, 'p Full ID')
        df_up.drop(columns=['p Full ID'], inplace=True)
        self.map_old_ids_to_new(df_up)
        #===========Temporal removal of participants===============
        #L = []
        #selection = ~df_up[self.merge_source].isin(L)
        #df_up = df_up[selection]
        #=========================================================
        if self.DOC in df_up.columns:
            df_up[self.DOC] = pd.to_datetime(df_up[self.DOC])
        vc = df_up[self.merge_source].value_counts()
        selection =  df_up[self.merge_source].value_counts().gt(1)
        if selection.any():
            print(vc[selection])
            print('Repetitions in the update are not allowed.')
            print('Repetitions will be removed (keep the first).')
            df_up.drop_duplicates(subset=self.merge_source,
                    keep='first', inplace=True)

        new_samples = ~df_up[self.merge_source].isin(self.df[self.merge_source])
        if new_samples.any():
            print('New samples:')
            print(df_up[new_samples])
            #raise ValueError('>>>Were you expecting new samples?')

        #print(df_up)
        print('Ready to merge')
        #Merge process >>>
        kind = 'original+'
        #This function call has now been moved outside
        #of this function in the parent object.
        #status_pre = self.compute_data_density()
        self.df = self.parent.merge_X_with_Y_and_return_Z(self.df,
                                                          df_up,
                                                          self.merge_source,
                                                          kind=kind)
        self.update_id_column()
        self.parent.SID_obj.check_df_dates_using_SID(self.df)
        print('The LSM file has been updated.')


    def plot_report(self):
        #How much data do we have?
        #This is the matrix density report.
        fname = 'LSM.xlsx'
        fname = os.path.join(self.dpath, fname)
        df = pd.read_excel(fname, sheet_name='report')
        df.rename(columns={'Count_current':'Count',
            'Empty_current':'Empty'}, inplace=True)
        df.dropna(subset=['Column'], axis=0, inplace=True)
        columns_to_plot=['Count','Empty']
        remove_rows = ['ID', 'Full ID', 'Date Collected']
        selection = df['Column'].isin(remove_rows)
        df = df[~selection].copy()
        ax = df.plot(x='Column', y=columns_to_plot, kind='bar')
        ax.set_xlabel('')
        print(df)

        ax2 = ax.twinx()
        n_labels = len(df)
        x = list(range(n_labels))
        ax2.plot(x,df['%_full_current'], 'bo',
                linestyle='-',
                markersize=5,
                alpha=0.6,
                label='%')
        plt.legend(loc='upper right')
        ax2.set_ylabel('% Processed')
        ax2.set_ylim([0, 100])

        fname = 'report.png'
        fname = os.path.join(self.dpath, fname)
        plt.tight_layout()
        plt.savefig(fname)

    def generate_L_format(self):
        #Jan 25 2023
        #Replicate the format in Lucas' table.
        #A=No infection before the DOC
        #B=Most recent infection before the DOC took place 
        #more than 3months 
        #C=Most recent infection before the DOC took place 
        #less than 3months 
        #self.parent.MPD_obj.add_site_column()

        v_date_cols = self.parent.LIS_obj.vaccine_date_cols
        v_type_cols = self.parent.LIS_obj.vaccine_type_cols
        inf_date_cols = self.parent.LIS_obj.positive_date_cols
        inf_type_cols = self.parent.LIS_obj.positive_type_cols
        index_to_i_status = {}
        print('Warning: Make sure the W file is updated.')
        fname  = 'W.xlsx'
        fname = os.path.join(self.parent.outputs_path, fname)
        df_w = pd.read_excel(fname)

        NPD  = 'Nearest pre-collection dose'
        #df_w[NPD] = 0
        df_w[NPD] = np.nan

        VD  = 'Vaccination date'
        df_w[VD] = np.nan

        DSD  = 'Days since dose'
        #df_w[DSD] = 0
        df_w[DSD] = np.nan

        MSD = 'Months since dose'
        df_w[MSD] = ''

        NPI = 'Nearest pre-collection infection'
        df_w[NPI] = np.nan

        IT = 'Infection type'
        df_w[IT] = np.nan

        DSI = 'Days since infection'
        df_w[DSI] = np.nan


        TIS = 'Ternary infection status'
        df_w[TIS] = ''

        BIS = 'Binary infection status'
        df_w[BIS] = ''

        DOC = self.DOC

        g_labels  = ['Full ID', 'Site',
                DOC, NPD, VD, DSD, MSD, NPI, IT, DSI, TIS, BIS]
        g_labels += self.numeric_columns

        rx_int = re.compile('[0-9]+')
        for index, row in df_w.iterrows():
            full_ID = row['Full ID']
            if pd.isnull(full_ID):
                continue
            #Vaccine calculation
            v_dates = row[v_date_cols]
            v_types = row[v_type_cols]

            #if v_dates.isnull().all():
                #continue

            #======Filter for 4 or more vaccines
            if v_dates.count() < 4:
                continue
            #======First 4 doses are mRNA and not bivalent
            v_type_selection = v_types.iloc[:4].isin(['Moderna','Pfizer'])
            if not v_type_selection.all():
                continue
            #======If 5 doses, check it is BModernaO
            if v_dates.count() == 5:
                if v_types.iloc[4] == 'BModernaO':
                    pass
                else:
                    continue
            selection = v_dates.notnull()
            v_dates = v_dates[selection]
            doc = row[DOC]
            deltas = (doc - v_dates) / np.timedelta64(1,'D')
            selection = deltas < 0
            deltas = deltas[~selection]
            if len(deltas) == 0:
                continue
            deltas = deltas.sort_values()
            v_dsd = deltas.iloc[0]
            df_w.loc[index, DSD] = v_dsd

            v_index = deltas.index[0]
            vaccine_date = row[v_index]
            vaccine_n = rx_int.search(v_index).group(0)
            vaccine_n = int(vaccine_n)
            df_w.loc[index, NPD] = vaccine_n
            df_w.loc[index, VD] = vaccine_date

            #Infection types (PCR and RAT)
            i_types = row[inf_type_cols]
            i_type_selection = i_types.isin(['PCR', 'RAT']).values
            i_types = i_types[i_type_selection]

            #Infection dates
            i_dates = row[inf_date_cols]
            i_dates = i_dates[i_type_selection]

            if i_dates.isnull().all():
                df_w.loc[index, TIS] = 'Not Inf'
                df_w.loc[index, BIS] = 'Not Infected'
                continue
            selection = i_dates.notnull()
            i_dates = i_dates[selection]
            deltas = (doc - i_dates) / np.timedelta64(1,'D')
            selection = deltas < 0
            deltas = deltas[~selection]
            if len(deltas) == 0:
                df_w.loc[index, TIS] = 'Not Inf'
                df_w.loc[index, BIS] = 'Not Infected'
                continue

            argmin = deltas.argmin()
            i_type = i_types.iloc[argmin]
            df_w.loc[index, IT] = i_type

            deltas = deltas.sort_values()
            i_index = deltas.index[0]
            infection_n = rx_int.search(i_index).group(0)
            infection_n = int(infection_n)
            df_w.loc[index, NPI] = infection_n

            nearest_infection_in_days = deltas.iloc[0]
            df_w.loc[index, DSI] = nearest_infection_in_days
            df_w.loc[index, BIS] = 'Infected'
            if nearest_infection_in_days < 90:
                df_w.loc[index, TIS] = 'Inf < 3mo'
            else:
                #Note that it is bigger than or equal to 90 days.
                df_w.loc[index, TIS] = 'Inf > 3mo'

        #Add site
        self.parent.MPD_obj.add_site_column(df_w)

        #Compute time labels
        max_days_since_dose = df_w[DSD].max()
        n_months_per_period  = 2
        days_per_month = 30
        days_per_period = days_per_month * n_months_per_period
        n_periods  = np.round(max_days_since_dose / days_per_period)
        days_upper = n_periods * days_per_period
        intervals = np.arange(0,days_upper+1, days_per_period, dtype=int)
        print(intervals)
        old = '0'
        time_labels = []
        for x in intervals[1:]:
            new = str(x // days_per_month)
            clone_new = new
            if len(clone_new) == 1:
                clone_new = ' ' + clone_new
            txt = old + '-' + clone_new
            old = new
            time_labels.append(txt)
        print(time_labels)
        n_time_labels = len(time_labels)

        #Create time classification
        bins = pd.cut(df_w[DSD], intervals, labels=time_labels)

        MSD = 'Months since dose'
        df_w[MSD] = bins

        df_w = df_w[g_labels]
        df_w.dropna(subset=['Full ID'], axis=0, inplace=True)
        df_w.dropna(subset=[NPD], axis=0, inplace=True)

        ####### Include metadata
        df_w = self.parent.create_df_with_ID_from_full_ID(df_w)
        #pd.merge(df_w, self.parent.df, on='ID', how='inner')
        #print(df_w)
        #return
        #######

        fname  = 'L_sans_metadata.xlsx'
        folder = 'Jessica_jan_25_2023'
        fname = os.path.join('..','requests',folder, fname)
        df_w.to_excel(fname, index=False)

    def generate_letter_to_AN_code_table(self):
        #Dec 23 2022
        #Alphanumeric code
        fname  = 'lookup_table_S_codes.xlsx'
        fname  = os.path.join(self.dpath, fname)
        df_t   = pd.read_excel(fname)
        print(df_t)
        txt = '[a-zA-Z ]+:(?P<ancode>[a-zA-Z0-9]+) - (?P<lcode>[A-Z]+)'
        rexp = re.compile(txt)
        for index, row in df_t.iterrows():
            h = row['Header']
            print(f'{h=}')
            obj = rexp.match(h)
            ancode = obj.group('ancode')
            lcode = obj.group('lcode')
            df_t.loc[index, 'Letter code'] = lcode
            df_t.loc[index, 'Alphanumeric code'] = ancode
        df_t.to_excel(fname, index=False)

    def generate_letter_to_AN_code_dict(self):
        #Dec 23 2022
        fname  = 'lookup_table_S_codes.xlsx'
        fname  = os.path.join(self.dpath, fname)
        df_t   = pd.read_excel(fname)
        for index, row in df_t.iterrows():
            ancode = row['Alphanumeric code']
            lcode  = row['Letter code']
            self.ancode_to_lcode[ancode] = lcode
            self.lcode_to_ancode[lcode] = ancode

    def check_full_id_format(self, df, col):
        #Modified to be applicable to any df and any given column.
        #user_length_set = set()
        #site_length_set = set()
        #Note that the time identifier is alphanumeric.
        #This function uses the "p Full ID" column to check the validity
        #of the format, and then creates the "standard" format in the
        #"Full ID" column.

        if 'Full ID' not in df.columns:
            df['Full ID'] = np.nan

        txt = '(?P<site>[0-9]{2})[-](?P<user>[0-9]{7})-(?P<time>[a-zA-Z0-9]+)'
        alpha_num_codes = ['Baseline', '9mo', '9moR', '12mo',
                           '12moR', '15mo', '15moR', '18mo',
                           '18moR', '21mo', '21moR', '3mo3',
                           '3mo3R', '6mo3', '6mo3R', '9mo3',
                           '9mo3R', '12mo3', '12mo3R', '15mo3',
                           '15mo3R', '3wk4', '3wk4R', '3mo4',
                           '3mo4R', '6mo4', '6mo4R', '9mo4',
                           '9mo4R', '12mo4', '12mo4R', '3wk5',
                           '3wk5R', '3mo5', '3mo5R']
        rexp = re.compile(txt)
        def process_id(txt):
            if isinstance(txt, str):
                obj = rexp.match(txt)
                if obj:
                    code = [obj.group('site'),
                            obj.group('user'),
                            obj.group('time')]
                    return code
                else:
                    print(txt)
                    raise ValueError(f'Unexpected format for {col=}')
            else:
                raise ValueError(f'Unexpected type for {txt=}')

        for index, row in df.iterrows():
            p_full_ID = row[col]
            code = process_id(p_full_ID)
            #The last element of the list 'code' has the time.
            doc_code = code[-1]
            if doc_code in alpha_num_codes:
                #The doc_code is alphanumeric and has to be converted.
                lcode = self.ancode_to_lcode[doc_code]
                #Get the time indicator from the whole label
                code[-1] = lcode
                full_ID  = '-'.join(code)
                df.loc[index, 'Full ID'] = full_ID
            else:
                df.loc[index, 'Full ID'] = p_full_ID

    def include_nucleocapsid_status(self):
        nuc_G = 'Nuc-IgG-100'
        nuc_G_s = 'Nuc-IgG-100 positive'
        nuc_A = 'Nuc-IgA-100'
        nuc_A_s = 'Nuc-IgA-100 positive'
        nuc_G_t = 0.547779865867836
        nuc_A_t = 0.577982139779995
        nuc_t   = [nuc_G_t, nuc_A_t]

        condition = nuc_G_s in self.df.columns and nuc_A_s in self.df.columns

        if not condition:
            #main labels
            labels = self.df.columns.to_list()[:4]
            labels += [nuc_G_s, nuc_A, nuc_A_s]
            labels += self.df.columns[5:].to_list()

        self.df[nuc_G_s] = nuc_G_t < self.df[nuc_G]
        self.df[nuc_A_s] = nuc_A_t < self.df[nuc_A]

        self.df[nuc_G_s] = self.df[nuc_G_s].where(self.df[nuc_G].notnull(),np.nan)
        self.df[nuc_A_s] = self.df[nuc_A_s].where(self.df[nuc_A].notnull(),np.nan)

        if not condition:
            self.df = self.df[labels].copy()

    def find_repeated_dates(self):
        #Jessica's request Feb 21 2023
        df_s = self.df.groupby('ID')
        for ID, df_g in df_s:
            vc = df_g['Date Collected'].value_counts()
            if vc.gt(1).any():
                vc = vc[vc.gt(1)]
                print(ID)
                print(vc)
                print('============')

    def nucleocapsid_stats(self):
        #Feb 23 2023
        inf_date_h = self.parent.LIS_obj.positive_date_cols
        boundary_date  = datetime.datetime(2022,1,1)
        nuc_G = 'Nuc-IgG-100'
        nuc_G_t = 0.547779865867836
        date_format = mpl.dates.DateFormatter('%b-%y')
        main_folder = 'Andrew_feb_23_2023'

        bio = nuc_G
        bio_t = nuc_G_t

        DOC = self.DOC

        before_boundary = {'Nuc_pos':[], 'Nuc_neg':[]}
        after_boundary = {'Nuc_pos':[], 'Nuc_neg':[]}

        N = self.parent.df.shape[0]
        for index_m, row_m in pbar(self.parent.df.iterrows(), total=N):
            inf_dates = row_m[inf_date_h]
            if inf_dates.count() == 0:
                #No infections, carry on.
                continue

            s = inf_dates.notnull()
            inf_dates = inf_dates[s]

            ID = row_m['ID']
            s = self.df['ID'] == ID
            df_s = self.df[s]

            n_bio_values = df_s[bio].count()
            if n_bio_values < 2:
                #No serology, carry on.
                continue
            #print(ID)
            s = df_s[bio].notnull()
            df_s = df_s[s].sort_values(DOC)
            iterator = df_s.iterrows()

            for (i1, r1), (i2, r2) in zip(iterator, iterator):

                d1 = r1[DOC]
                d2 = r2[DOC]

                v1 = r1[bio]
                v2 = r2[bio]

                for col_name, inf_date in inf_dates.items():

                    if d1 <= inf_date and inf_date <= d2:
                        #We have an infection in between.
                        dt = (d2 - inf_date) / np.timedelta64(1,'D')

                        if v1 < bio_t and bio_t < v2:
                            #We crossed
                            nuc_status = 'Nuc_pos'
                        else:
                            nuc_status = 'Nuc_neg'

                        fig, ax = plt.subplots()
                        ax.plot([d1,d2],[v1,v2],'bo-')
                        ax.axvline(inf_date, color='red', linewidth=3)
                        ax.axhline(bio_t, color='grey', linewidth=3)
                        ax.set_ylabel(bio)
                        ax.xaxis.set_major_formatter(date_format)

                        if d2 <= boundary_date:
                            bd_status = 'before'
                            before_boundary[nuc_status].append(ID)
                        else:
                            bd_status = 'after'
                            after_boundary[nuc_status].append(ID)

                        txt = '(' + bd_status + ')  '
                        txt += ID
                        txt += '    $\Delta t=$' + str(int(dt)) + ' days'
                        ax.set_title(txt)

                        fname = ID + '.png'

                        folder = os.path.join(self.parent.requests_path,
                                main_folder,
                                bd_status,
                                nuc_status)

                        if not os.path.exists(folder):
                            os.makedirs(folder)

                        fname = os.path.join(folder, fname)
                        fig.savefig(fname)
                        plt.close('all')



                        #We leave the loop since we just need one infection.
                        break

        #print('Before', boundary_date)

    def check_vaccine_labels(self):
        txt = '(?P<site>[0-9]{2})[-](?P<user>[0-9]{7})-(?P<time>[a-zA-Z0-9]+)'
        extract_letter = re.compile(txt)
        time_exp_to_mult_in_days = {'mo': 30, 'wk': 7}
        txt = '(?P<tpost>[0-9]+)(?P<time_exp>[a-z]+)(?P<dose>[0-9]+)'
        txt += '(?P<repeat>R)?'
        extract_time_and_dose = re.compile(txt)
        s = slice('ID','Date Collected')
        df = self.df.loc[:,s].copy()
        extract_number = re.compile('[0-9]+')

        vac_dates_h = self.parent.LIS_obj.vaccine_date_cols

        df['Short code'] = np.nan
        df['Long code'] = np.nan
        df['Dose #'] = np.nan
        df['Dose date'] = np.nan
        df['Time post-dose (real)'] = np.nan
        df['Time post-dose (expected)'] = np.nan
        df['|Delta T| (current)'] = np.nan
        df['Recommended dose #'] = np.nan
        df['Recommended dose date'] = np.nan
        df['Time post-rec. dose'] = np.nan
        df['Recommended Short code'] = np.nan
        df['Time post-rec. dose (expected)'] = np.nan
        df['|Delta T| (recalculated)'] = np.nan
        df['Improvement in days'] = np.nan
        df['Comment'] = np.nan
        df['Recommendation exists?'] = np.nan
        df['Warning'] = np.nan

        N = df.shape[0]

        for index_s, row_s in pbar(df.iterrows(), total=N):
            full_ID = row_s['Full ID']
            ID = row_s['ID']
            obj = extract_letter.match(full_ID)
            if obj:
                letter = obj.group('time')
            else:
                raise ValueError(full_ID)

            #Get information for this sample
            s = self.serology_codes['Letter code'] == letter
            index_c = s[s].index[0]
            ancode = self.serology_codes.loc[index_c, 'Alphanumeric code']
            dose = self.serology_codes.loc[index_c, 'Dose']
            if pd.notnull(dose):
                #print('===========================')
                #print(f'{full_ID=}')
                #print(letter, '-->', ancode)
                dose = int(dose)
                temp = self.serology_codes.loc[index_c, 'Post-dose days']
                days_post_dose_expected = int(temp)
                repeat_value = self.serology_codes.loc[index_c, 'Repeat']
                repeat_value = int(repeat_value)
                doc = row_s[self.DOC]
                vac_date_h = 'Vaccine Date ' + str(dose)
                s = self.parent.df['ID'] == ID
                index_m = s[s].index[0]
                vac_date = self.parent.df.loc[index_m, vac_date_h]
                days_post_dose_real = (doc - vac_date) / np.timedelta64(1,'D')
                #print(f'{days_post_dose_t=}')
                #print(f'{days_post_dose_e=}')
                delta = np.abs(days_post_dose_real - days_post_dose_expected)

                df.loc[index_s,'Short code'] = letter
                df.loc[index_s,'Long code'] = ancode
                df.loc[index_s,'Dose #'] = dose
                df.loc[index_s,'Dose date'] = vac_date
                df.loc[index_s,'Time post-dose (real)'] = days_post_dose_real
                df.loc[index_s,'Time post-dose (expected)'] = days_post_dose_expected
                df.loc[index_s,'|Delta T| (current)'] = delta

                #Optimal candidate
                vac_dates = self.parent.df.loc[index_m, vac_dates_h]
                s = vac_dates.notnull()

                if not s.any():
                    df.loc[index_s,'Comment'] = 'No vaccination'
                    df.loc[index_s,'Warning'] = True
                    continue

                vac_dates = vac_dates[s]
                delta_dates = doc - vac_dates
                delta_dates = delta_dates.dt.days
                #print(delta_dates)
                delta_dates = delta_dates[delta_dates.gt(0)]
                #print(delta_dates)
                sorted_deltas = delta_dates.sort_values()
                rec_dose = sorted_deltas.index[0]
                calculated_diff = int(sorted_deltas.iloc[0])
                obj = extract_number.search(rec_dose)
                if obj:
                    rec_dose = int(obj.group(0))
                else:
                    raise ValueError('Unable to extract dose number')

                df.loc[index_s, 'Recommended dose #']  = rec_dose

                rec_vac_date_h = 'Vaccine Date ' + str(rec_dose)
                rec_vac_date = self.parent.df.loc[index_m, rec_vac_date_h]
                df.loc[index_s, 'Recommended dose date']  = rec_vac_date

                df.loc[index_s, 'Time post-rec. dose'] = calculated_diff
                #print(f'{rec_dose=}')
                #print(f'{repeat_value=}')

                if dose != rec_dose:
                    df.loc[index_s,'Comment'] = 'Potentially wrong dose #'

                if rec_dose < 3:
                    df.loc[index_s,'Warning'] = True
                    continue

                s =  self.serology_codes['Dose'] == rec_dose
                s &= self.serology_codes['Repeat'] == repeat_value
                available_post_days = self.serology_codes.loc[s,'Post-dose days']
                deltas = np.abs(available_post_days - calculated_diff)
                deltas = deltas.sort_values()
                index_opt = deltas.index[0]
                #print(deltas)
                #print(index_opt)
                rec_letter = self.serology_codes.loc[index_opt, 'Letter code']
                df.loc[index_s,'Recommended Short code'] = rec_letter
                expected_time = self.serology_codes.loc[index_opt, 'Post-dose days']
                expected_time = int(expected_time)
                df.loc[index_s, 'Time post-rec. dose (expected)'] = expected_time
                delta = np.abs(expected_time - calculated_diff)
                df.loc[index_s,'|Delta T| (recalculated)'] = delta

                a = df.loc[index_s,'|Delta T| (current)']
                b = df.loc[index_s,'|Delta T| (recalculated)']
                df.loc[index_s,'Improvement in days'] = a-b

                if rec_letter != letter:
                    df.loc[index_s,'Warning'] = True
                    if dose == rec_dose:
                        df.loc[index_s,'Comment'] = 'Potentially better letter code'
                    new_full_ID = ID + '-' + rec_letter
                    s = self.df['Full ID'] == new_full_ID
                    if s.any():
                        df.loc[index_s,'Recommendation exists?'] = True


        #print(df)
        folder = 'Megan_feb_24_2023'
        fname = 'serology_label_verification_feb_25_2023.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df.to_excel(fname, index=False)




