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


# <b> Section: Master_Participant_Data </b>

class LTCSerologyMaster:
    def __init__(self, dpath, parent=None):
        self.parent       = None
        self.df           = None
        self.delta_report = None
        self.dpath        = dpath
        self.backups_path = os.path.join('..', 'backups')
        self.merge_source = 'Full ID'
        self.merge_column = 'ID'
        self.ID_columns   = [self.merge_source,
                self.merge_column]
        self.DOC          = 'Date Collected'
        self.non_numeric_columns = []
        self.numeric_columns     = []

        self.load_LSM_file()


        if parent:
            self.parent = parent
            print('LSM class initialization from Manager.')
        else:
            raise ValueError('Parent object is unavailable.')

    def load_LSM_file(self):
        #Note that the LSM file already has compact format, i.e.,
        #the headers are simple.
        fname = 'LSM.xlsx'
        fname = os.path.join(self.dpath, fname)
        #Read the Excel file containing the data
        self.df = pd.read_excel(fname)
        self.non_numeric_columns = self.ID_columns + [self.DOC]
        for column in self.df.columns:
            if column not in self.non_numeric_columns:
                self.numeric_columns.append(column)

        print('LSM class has been initialized with LSM file.')


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
        self.backup_the_LSM_file()

        print('Writing the LSM file to Excel.')

        fname = 'LSM.xlsx'
        fname = os.path.join(self.dpath, fname)
        with pd.ExcelWriter(fname) as writer:
            self.df.to_excel(writer,
                    sheet_name = 'data', index = False)
            self.delta_report.to_excel(writer,
                    sheet_name = 'report', index = False)

        print('The LSM file has been written to Excel.')

    def merge_serology_update(self, df_up):
        #Updated on Oct 31, 2022
        print('===================Work======')
        df_up.replace('NT', np.nan, inplace=True)
        relevant_proteins = ['Spike', 'RBD', 'Nuc']
        relevant_Igs      = ['IgG', 'IgA', 'IgM' ]
        rexp_n = re.compile('/[ ]*(?P<dilution>[0-9]+)')
        rexp_c = re.compile('[0-9]{2}[-][0-9]{7}[-][a-zA-Z]{1,2}')
        col_indices = {}
        exit_iterrows_flag = False
        row_start = -1
        id_col_index = -1
        list_of_proteins = []
        list_of_Igs = []
        list_of_dilutions = []
        for index, row in df_up.iterrows():
            if exit_iterrows_flag:
                break
            for col, item in row.items():
                if isinstance(item, str):
                    #Check if the item is an ID
                    obj = rexp_c.search(item)
                    if obj:
                        #print('Data start at row:', index)
                        id_col_index = col
                        row_start = index
                        exit_iterrows_flag = True
                        break
                    for protein in relevant_proteins:
                        if protein.lower() in item.lower():
                            col_indices[col] = None
                            list_of_proteins.append(protein)
                            break
                    for Ig in relevant_Igs:
                        if Ig.lower() in item.lower():
                            col_indices[col] = None
                            list_of_Igs.append(Ig)
                            break
                    #Check if the item is a dilution
                    obj = rexp_n.search(item)
                    if obj:
                        dilution = obj.group('dilution')
                        col_indices[col] = None
                        list_of_dilutions.append(dilution)
        #Form headers
        for k, p, Ig, dil in zip(col_indices.keys(),
                                 list_of_proteins,
                                 list_of_Igs,
                                 list_of_dilutions):
            s = '-'.join([p,Ig,dil])
            col_indices[k] = s

        #Full ID
        merge_at_column = self.merge_source
        #Set in the dictionary the mapping:
        #id_col_index -> merge_at_column
        col_indices[id_col_index] = merge_at_column
        print(col_indices)
        df_up.rename(columns = col_indices, inplace = True)
        def is_id(txt):
            if txt is np.nan:
                return txt
            obj = rexp_c.search(txt)
            if obj:
                return obj.group(0)
            else:
                return np.nan
        df_up[merge_at_column] = df_up[merge_at_column].apply(is_id)
        df_up.dropna(subset=merge_at_column, axis=0, inplace=True)
        #Remove individuals with "E" type label.
        self.remap_E_type_individuals(df_up)
        print('Ready to merge')
        #Merge process >>>
        #The update has a higher priority than the original data.
        kind = 'update+'
        self.df = self.parent.merge_X_with_Y_and_return_Z(self.df,
                                                          df_up,
                                                          merge_at_column,
                                                          kind=kind)
        self.update_id_column()
        print('End of updating the LSM file.')

    def remap_E_type_individuals(self, df_up):
        #Careful with the "E" type individuals.
        fname  = 'remapping_list.xlsx'
        fname = os.path.join(self.dpath, fname)
        df_re = pd.read_excel(fname)
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
        self.remap_E_type_individuals(df_up)
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
        #Nov 24 2022
        regexp = re.compile('[0-9]{2}[-][0-9]{7}[-](?P<collection>[a-zA-Z]{1,2})')
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

    def compute_data_density(self):
        col_to_n_data = {'Column':[], 'Count':[], 'Empty':[], '%_full':[]}
        n_rows = len(self.df)
        for column in self.df.columns:
            n_data = self.df[column].count()
            col_to_n_data['Column'].append(column)
            col_to_n_data['Count'].append(n_data)
            col_to_n_data['Empty'].append(n_rows-n_data)
            p = n_data/n_rows*100
            col_to_n_data['%_full'].append(p)
        df = pd.DataFrame(col_to_n_data)
        return (df, n_rows)

    def monotonic_increment_check(self, pre, post):
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

    def direct_serology_update_with_headers(self):
        fname  = 'NazyAbData_2022_11_14.xlsx'
        print(f'Working with {fname=}')
        folder = 'Jessica_dec_07_2022'
        fname = os.path.join(self.parent.requests_path,
                folder, fname)
        df_up = pd.read_excel(fname, sheet_name='Sept 28 2022')
        print(f'The update has {len(df_up)} rows.')
        self.remap_E_type_individuals(df_up)
        df_up.replace('.', np.nan, inplace=True)
        df_up.replace('n/a', np.nan, inplace=True)
        df_up.replace('N/A', np.nan, inplace=True)
        df_up.replace('#N/A', np.nan, inplace=True)
        df_up.replace('retest on next plate', np.nan, inplace=True)
        df_up.replace('NT', np.nan, inplace=True)
        df_up.dropna(axis=0, how='all', inplace=True)
        df_up[self.merge_source] = df_up[self.merge_source].str.replace(' ','')
        #=========== Removal of cutoff===============
        selection = df_up[self.merge_source].str.lower().str.contains('cutoff')
        df_up = df_up[~selection]
        #===========ID verification===============
        rexp_c = re.compile('[0-9]{2}[-][0-9]{7}[-][A-Z]{1,2}')
        def is_a_valid_id(txt):
            obj = rexp_c.search(txt)
            if obj:
                return obj.group(0)
            else:
                print(txt)
                raise ValueError('Not an ID')
        df_up[self.merge_source] = df_up[self.merge_source].map(is_a_valid_id)
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
        #The update has a higher priority than the original data.
        kind = 'original+'
        status_pre = self.compute_data_density()
        self.df = self.parent.merge_X_with_Y_and_return_Z(self.df,
                                                          df_up,
                                                          self.merge_source,
                                                          kind=kind)
        self.update_id_column()
        self.parent.SID_obj.check_df_dates_using_SID(self.df)
        status_post = self.compute_data_density()
        self.monotonic_increment_check(status_pre, status_post)
        print('The LSM file has been updated.')



