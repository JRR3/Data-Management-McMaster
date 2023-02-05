#JRR @ McMaster University
#Update: 10-Oct-2022
import numpy as np
import pandas as pd
#pd.options.plotting.backend = 'plotly'
#import plotly.express as pxp
import os
import re
import networkx as nx
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
#mpl.rcParams['font.family'] = 'monospace'
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
# <b> Section: Reporter </b>
# This class takes care of all the visualization
# tasks related to the M file.

class Reporter:
    def __init__(self, path, parent=None):
        #In this class we deal with the infections
        #and the vaccines.

        self.parent = None
        self.dpath = path
        self.requests_path = os.path.join('..', 'requests')


        if parent:
            self.parent = parent
            print('Reporter class initialization from Manager.')
        else:
            pass

    def plot_n_infections_pie_chart(self):
        pd_cols = self.parent.LIS_obj.positive_date_cols
        inf_vec = np.zeros(len(pd_cols)+1)
        for index, row in self.parent.df.iterrows():
            p_dates = row[pd_cols]
            if p_dates.isnull().all():
                inf_vec[0] += 1
            else:
                value = p_dates.notnull().value_counts().loc[True]
                inf_vec[value] += 1
        possible_n_inf = np.arange(len(pd_cols)+1, dtype=int)
        df = pd.DataFrame({'# of infections': possible_n_inf, 'cases': inf_vec})
        fig = pxp.pie(df,
                      names='# of infections',
                      values='cases',
                      title='Number of infections')
        fig.update_layout( font_size=20)
        fig.update_layout(hoverlabel={'font_size':20})
        fname = 'n_of_infections_pie.html'
        fname = os.path.join(self.dpath, fname)
        fig.write_html(fname)

    def plot_vaccine_choice(self):
        cols = self.parent.LIS_obj.vaccine_type_cols
        df = self.parent.df[cols].copy()
        L = []
        for k,col in enumerate(cols):
            x = df.groupby(col)[col].count()
            x.index.name = None
            x = x.to_frame().reset_index()
            x['Vaccine #'] = k+1
            x.rename(columns={col:'Counts', 'index':'Brand'}, inplace=True)
            L.append(x)
        df = pd.concat(L, axis=0)
        print(df)
        fig = pxp.bar(df,
                      x='Vaccine #',
                      y='Counts',
                      color='Brand',
                      color_discrete_sequence=pxp.colors.qualitative.Plotly,
                      category_orders={'Brand':['Pfizer',
                                                'Moderna',
                                                'BModernaO',
                                                'AstraZeneca',
                                                'BPfizerO',
                                                'COVISHIELD']},
                      title='Vaccine Type')
        fig.update_layout( font_size=20)
        fig.update_layout(hoverlabel={'font_size':20})
        fname = 'vaccine_type_counts.html'
        fname = os.path.join(self.dpath, fname)
        fig.write_html(fname)

        fig = pxp.histogram(df,
                      x='Vaccine #',
                      y='Counts',
                      color='Brand',
                      barnorm='percent',
                      color_discrete_sequence=pxp.colors.qualitative.Plotly,
                      category_orders={'Brand':['Pfizer',
                                                'Moderna',
                                                'BModernaO',
                                                'AstraZeneca',
                                                'BPfizerO',
                                                'COVISHIELD']},
                      title='Vaccine Type')
        fig.update_layout( font_size=20, yaxis_title='%')
        fig.update_layout(hoverlabel={'font_size':20})
        fname = 'vaccine_type_percent.html'
        fname = os.path.join(self.dpath, fname)
        fig.write_html(fname)

    def plot_infections_by_dates(self, freq = 15):
        cols = self.parent.LIS_obj.positive_date_cols
        df = self.parent.df[cols].copy()
        dt = df.values.ravel()
        dt = dt[~pd.isnull(dt)]
        ps = pd.Series(index=dt, data=1)
        freq_str = str(freq) + 'D'
        ps  = ps.resample(freq_str).count()
        df  = ps.to_frame().reset_index()
        df.rename(columns={0:'Infections', 'index':'Date'}, inplace=True)
        print(df)
        fig = pxp.line(df,
                       x='Date',
                       y='Infections',
                       title='Infection time series-' + freq_str)
        fig.update_traces(line_width=5)
        fig.update_layout( font_size=20)
        fig.update_layout(hoverlabel={'font_size':20})
        fname = 'infection_time_series' + freq_str + '.html'
        fname = os.path.join(self.dpath, fname)
        fig.write_html(fname)

    def generate_Ab_post_vaccination_df(self):
        #Generate Data Frame for Ab levels after vaccination.
        #A plot is also generated.

        lsm_req = 'RBD-IgG-100'
        had_infection = False
        had_inf_str   = str(had_infection)
        vac_date_cols = self.parent.LIS_obj.vaccine_date_cols
        max_n_vac     = len(vac_date_cols)
        max_bin_index = {}
        #Number of days in a bin.
        coarseness = 15
        #Note that the only reason we use a dictionary is to
        #verify that the computations are correct. For
        #practical purposes, a list would be sufficient.
        m_dc = {}
        for m_index, m_row in self.parent.df.iterrows():
            ID = m_row['ID']
            last_inf = m_row['Last infection']
            if pd.notnull(last_inf) == had_infection:
                pass
            else:
                continue
            lsm_selector = self.parent.LSM_obj.df['ID'] == ID
            #print('----------')
            df_slice = self.parent.LSM_obj.df.loc[lsm_selector,:]
            #print('Len:', len(df_slice))
            #We iterate over all the samples for a given individual.
            #Can 2 or more samples fall in the same interval? Yes.
            #We keep the most recent one.
            time_dc = {}
            for lsm_index, lsm_row in df_slice.iterrows():
                value = lsm_row[lsm_req]
                if pd.isnull(value):
                    #We do not have this serology value.
                    #Keep looking.
                    #Move on to next sample.
                    continue
                doc = lsm_row['Date Collected']
                had_vaccines = m_row[vac_date_cols].notnull().any()
                if ~had_vaccines:
                    #Without vaccines we cannot classify the doc.
                    continue
                #Get the first vaccination date
                vac_date = m_row[vac_date_cols[0]]
                if pd.isnull(vac_date):
                    raise ValueError('No first vaccine for this individual.')
                #We use a variable called v_index, which denotes the
                #vaccine index. Zero means before the first vaccination
                #date, i.e., the 0-th period.
                #One means before the second vaccination date, i.e.,
                #the 1-st period, and so on.
                v_index = 0
                if doc < vac_date:
                    #Sample was taken before the first vaccine.
                    #No need to use a special bin.
                    bin_index = 0
                    delta = (doc - vac_date) / np.timedelta64(1,'D')
                    if (v_index,bin_index) not in time_dc:
                        time_dc[(v_index, bin_index)] = []
                    time_dc[(v_index,
                             bin_index)].append((value,
                                                 vac_date,
                                                 doc,
                                                 delta))
                    continue
                #At this point we know that the doc is >= to the first
                #vaccination date, which we will call the previous
                #vaccination date, i.e., prev_vac_date.
                prev_vac_date = vac_date
                #Now we iterate over the remaining vaccination dates.
                found_upper_bound = False
                is_null_flag      = False
                for vd_col in vac_date_cols[1:]:
                    #We add one to the index because we are
                    #actually starting from the 
                    #second vaccination date.
                    v_index += 1
                    vac_date = m_row[vd_col]
                    if pd.isnull(vac_date):
                        #We assume vaccination dates are contiguous.
                        #If it is null, we know that the last
                        #vaccination date is prev_vac_date.
                        #No need to continue iterating over the
                        #vac_date_cols[1:].
                        is_null_flag = True
                        break
                    if doc < vac_date:
                        found_upper_bound = True
                        #At this point we know that the
                        #doc happened before the current
                        #vaccination date. This means we have
                        #to count from the prev_vac_date to
                        #the doc.
                        delta = (doc-prev_vac_date) / np.timedelta64(1,'D')
                        delta = np.abs(delta)
                        bin_index = delta // coarseness
                        mbi = max_bin_index.get(v_index, -1)
                        if mbi < bin_index:
                            max_bin_index[v_index] = bin_index
                        if (v_index,bin_index) not in time_dc:
                            time_dc[(v_index, bin_index)] = []
                        time_dc[(v_index,
                                 bin_index)].append((value,
                                                     prev_vac_date,
                                                     doc,
                                                     delta))
                        #No need to continue iterating over vaccine dates.
                        break
                    #If we are here, it means the doc is >= to the
                    #current vaccination date, which will become the
                    #previous vaccination date for the next iteration.
                    prev_vac_date = vac_date
                if found_upper_bound:
                    #If we found an upper bound, it means
                    #we already included this information in
                    #the dictionary.
                    #Keey iterating over the remaining samples.
                    continue
                else:
                    if is_null_flag:
                        #We are still within the vaccine dates.
                        #We reached this point because one vaccine
                        #date was empty. We know that the doc happened
                        #at or after the prev_vac_date.
                        pass
                    else:
                        #The doc is after all vaccine dates.
                        #This means that the vaccine index has to
                        #be increased by 1.
                        v_index += 1
                    delta = (doc-prev_vac_date) / np.timedelta64(1,'D')
                    delta = np.abs(delta)
                    bin_index = delta // coarseness
                    mbi = max_bin_index.get(v_index, -1)
                    if mbi < bin_index:
                        max_bin_index[v_index] = bin_index
                    if (v_index,bin_index) not in time_dc:
                        time_dc[(v_index, bin_index)] = []
                    time_dc[(v_index,
                             bin_index)].append((value,
                                                 prev_vac_date,
                                                 doc,
                                                 delta))

            #End of iterating over all the samples of an individual.
            #If the dictionary is nonempty, we store it.
            if 0 < len(time_dc):
                m_dc[ID] = time_dc
        #Largest vaccine index for the max_bin_index dictionary.
        max_key = 0
        for key, _ in max_bin_index.items():
            if max_key < key:
                max_key = key
        L = [1]
        for k in range(1,max_key+1):
            v_index = k
            n_bins  = max_bin_index[v_index] + 1
            L.append(n_bins)
        cols_per_v_index = np.array(L, dtype=int)
        print(cols_per_v_index)
        x = np.cumsum(cols_per_v_index)
        v_index_to_increment = np.concatenate(([0], x[:-1]))
        n_cols = cols_per_v_index.sum()
        n_rows = len(m_dc)
        m = np.zeros((n_rows,n_cols), dtype=float)
        #Generate columns
        L = []
        L.append('Pre-Vac')
        for v_index in range(max_key+1):
            if v_index == 0:
                continue
            base = 'V' + str(v_index) + '+'
            start = 0
            for k in range(cols_per_v_index[v_index]):
                txt = base
                end = start + coarseness - 1
                txt += '(' + str(start) + ',' + str(end) + ')'
                txt += 'D'
                start = end + 1
                L.append(txt)
        column_names = L.copy()
        #(value, prev_vac_date, doc, delta)
        doc_index = 2
        #Populate the matrix m.
        for row, (ID, ID_dc) in enumerate(m_dc.items()):
            for (v_index, bin_index), L in ID_dc.items():
                increment = v_index_to_increment[v_index]
                col = increment + int(bin_index)
                stored_doc = datetime.datetime(1900,1,1)
                most_recent_index = -1
                #Iterate over the elements of the list
                #containing the Ig values and dates of
                #collection. 
                #We want the most recent DOC.
                for k, vec in enumerate(L):
                    doc= vec[doc_index]
                    if stored_doc < doc:
                        most_recent_index = k
                        stored_doc = doc
                Ig_value = L[most_recent_index][0]
                m[row,col] = Ig_value

        df = pd.DataFrame(m, columns=column_names, index = m_dc.keys())
        df.replace(0., np.nan, inplace=True)
        means = df.mean(axis=0)
        df = means.to_frame()
        if had_infection:
            col_name = 'I+'
        else:
            col_name = 'I-'
        df = df.reset_index().rename(columns={'index':'Bin', 0:col_name})
        infection_status= 'Had infection:' + had_inf_str
        fig = pxp.line(df, x='Bin', y=col_name, title=infection_status)
        fig.update_traces(line_width=5)
        fig.update_layout( font_size=20)
        fig.update_layout(hoverlabel={'font_size':20})
        txt = lsm_req
        txt+= '_trend_'
        txt+= 'infection_is_' + had_inf_str
        plot_name = txt + '.html'
        fname = os.path.join(self.dpath, plot_name)
        fig.write_html(fname)

        df_name = txt + '.xlsx'
        fname = os.path.join(self.dpath, df_name)
        df.to_excel(fname, index=False)

    def merge_and_plot_Ab_data(self):
        #Unfinished
        f = 'RBD-IgG-100_trend_infection_is_False.xlsx'
        fname = os.path.join(self.dpath, f)
        df_1 = pd.read_excel(fname)
        f = 'RBD-IgG-100_trend_infection_is_True.xlsx'
        fname = os.path.join(self.dpath, f)
        df_2 = pd.read_excel(fname)
        m = pd.merge(df_1, df_2, on='Bin', how='outer')
        #print(m)
        fig = pxp.line(m, x='Bin', y=['Ab(I-)', 'Ab(I+)'])
        fig.update_traces(line_width=5)
        fig.update_layout(font_size=20)
        fig.update_layout(hoverlabel={'font_size':20})
        fig.update_layout(legend={'title':'Had an infection?'})
        plot_name = 'merged_Ab_data.html'
        fname = os.path.join(self.dpath, plot_name)
        fig.write_html(fname)



    def plot_serology_slope_progression(self):
        fpure = 'infection_dates_slope.xlsx'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        df    = pd.read_excel(fname)
        n_rows = len(df)
        n_infections = np.sort(df['# infections'].unique())
        case_counter = 0
        for n in n_infections:
            if n == 1:
                continue
            selection  = df['# infections'] == n
            selection &= df['S: Has slopes?'] == True
            selection &= df['S: Had 0 days?'] == False
            df_s = df.loc[selection, :]
            if len(df_s) == 0:
                continue
            unique_ids = df_s['ID'].unique()
            for ID in unique_ids:
                selection = df_s['ID'] == ID
                plt.close('all')
                fig, ax = plt.subplots()
                df_p = df_s.loc[selection,:]
                if len(df_p) < 2:
                    #Not enough data to plot a progression
                    continue
                #Avoid collinearities
                if df_p['S: Date before'].value_counts().gt(1).any():
                    continue
                if df_p['S: Date after'].value_counts().gt(1).any():
                    continue
                print('===============')
                print(f'{ID=}')
                case_counter += 1
                for k, (index, row) in enumerate(df_p.iterrows()):
                    self.serology_line_from_row_to_plot(row,
                                                        ax,
                                                        case=k,
                                                        use_label=False)
                self.save_slope_plot_for_ID(fig,
                                            ax,
                                            row,
                                            ID,
                                            df_p,
                                            same_folder=True)
        print(f'{case_counter=}')

    def save_slope_plot_for_ID(self,
                               fig,
                               ax,
                               row,
                               ID,
                               df,
                               same_folder=False):
        folder_1= 'Braeden_oct_20_2022'
        folder_2= 'participants'
        if same_folder:
            dpath = os.path.join(self.parent.requests_path,
                                 folder_1,
                                 folder_2,
                                 '00_noncollinear')
        else:
            dpath = os.path.join(self.parent.requests_path,
                                 folder_1,
                                 folder_2,
                                 ID)
        if not os.path.exists(dpath):
            os.makedirs(dpath)
        #Add redundant line to create labels
        self.serology_line_from_row_to_plot(row,
                                            ax,
                                            case=-1,
                                            use_label=True)
        rx = re.compile('[0-9]+')
        events = df['Infection event']
        L = []
        for index, event in events.items():
            inf_n = rx.search(event).group(0)
            L.append(inf_n)
        inf_seq = ID + ' infections: ' + ', '.join(L)
        ax.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b-%y"))
        ax.set_ylabel('OD')
        plt.xticks(rotation=45)
        plt.legend(loc='best')
        plt.title(inf_seq)
        fpure = ID + '_slope_progression.png'
        fname = os.path.join(dpath, fpure)
        fig.savefig(fname)

    def serology_line_from_row_to_plot(self,
                                       row,
                                       ax,
                                       case=-1,
                                       use_label=False):
        #The case is to select a marker type for the plot.
        mk_size = 4
        #event = row['Infection event']
        #rx = re.compile('[0-9]+')
        #inf_n = rx.search(event).group(0)
        di = row['Infection date']
        d1 = row['S: Date before']
        d2 = row['S: Date after']
        dt_1 = row['S: Days before']
        dt_2 = row['S: Days after']
        IgG1 = row['S: Nuc-IgG-100 before']
        IgG2 = row['S: Nuc-IgG-100 after']
        IgA1 = row['S: Nuc-IgA-100 before']
        IgA2 = row['S: Nuc-IgA-100 after']
        IgG_slope = row['S: Slope IgG']
        IgA_slope = row['S: Slope IgA']
        if not use_label:
            print(f'{IgG_slope=:+.2E}')
            print(f'{IgA_slope=:+.2E}')
        IgG_mid = IgG1 + dt_1 * IgG_slope
        IgA_mid = IgA1 + dt_1 * IgA_slope
        IgG25 = IgG1 + dt_1 / 2 * IgG_slope
        IgG75 = IgG1 + (dt_1 + dt_2 / 2) * IgG_slope
        IgA25 = IgA1 + dt_1 / 2 * IgA_slope
        IgA75 = IgA1 + (dt_1 + dt_2 / 2) * IgA_slope
        if use_label:
            ax.plot([d1,d2], [IgG1, IgG2], 'b-', label='IgG')
            ax.plot([d1,d2], [IgA1, IgA2], 'r-', label='IgA')
        else:
            ax.plot([d1,d2], [IgG1, IgG2], 'b-')
            ax.plot([d1,d2], [IgA1, IgA2], 'r-')
        ax.plot([di], [IgG_mid], 'ko', markersize=mk_size)
        ax.plot([di], [IgA_mid], 'ko', markersize=mk_size)
        date25 = di - pd.DateOffset(days = dt_1 / 2)
        date75 = di + pd.DateOffset(days = dt_2 / 2)
        t1 = str(int(dt_1))
        t2 = str(int(dt_2))
        factor = 0.025
        mplus  = 1+factor
        mminus = 1-factor
        ax.text(date25, IgG25*mplus, t1)
        ax.text(date75, IgG75*mplus, t2)
        ax.text(date25, IgA25*mminus, t1)
        ax.text(date75, IgA75*mminus, t2)
        markers = ['P', 'o', 'x', 'd', '*']
        mc = ['k', 'g', 'c', 'm', 'y']
        mk_size_2 = 6
        if -1 < case:
            #Plot markers for the start and end points.
            IgGm = 'b' + markers[case]
            IgAm = 'r' + markers[case]
            ax.plot([d1], [IgG1], IgGm, markersize=mk_size_2, markerfacecolor='None')
            ax.plot([d2], [IgG2], IgGm, markersize=mk_size_2, markerfacecolor='None')
            ax.plot([d1], [IgA1], IgAm, markersize=mk_size_2, markerfacecolor='None')
            ax.plot([d2], [IgA2], IgAm, markersize=mk_size_2, markerfacecolor='None')





    def plot_serology_one_Ig_from_df(self, Ig='G', max_n_inf=5):
        fpure = 'infection_dates_slope.xlsx'
        folder= 'Braeden_oct_20_2022'
        threshold = {'G':(0.54, 0.55), 'A':(0.57, 0.58)}
        Ig_color = {'G':['blue'], 'A':['red']}
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        df    = pd.read_excel(fname)
        fig, ax = plt.subplots()
        mk_size = 4
        Ig_before_L = 'S: Nuc-Ig' + Ig + '-100 before'
        Ig_after_L  = 'S: Nuc-Ig' + Ig + '-100 after'
        Ig_slope_L  = 'S: Slope Ig' + Ig
        Ig_ratio_L  = 'S: Ratio Ig' + Ig
        zero_days_L = 'S: Had 0 days?'
        method_L    = 'Method'
        n_inf_L     = '# infections'
        thresh      = threshold[Ig]
        selection  = df[method_L] == 'PCR'
        selection &= df[n_inf_L] <= max_n_inf
        selection &= df[zero_days_L] == False
        selection &= df[Ig_before_L] < thresh[0]
        selection &= df[Ig_after_L] > thresh[1]
        df = df.loc[selection,:]
        n_rows = len(df)
        line_col = Ig_color[Ig][0]

        for k, (index, row) in enumerate(df.iterrows()):
            di = row['Infection date']
            d1 = row['S: Date before']
            d2 = row['S: Date after']
            dt_1 = row['S: Days before']
            Ig1 = row[Ig_before_L]
            Ig2 = row[Ig_after_L]
            Ig_slope = row[Ig_slope_L]
            Ig_mid = Ig1 + dt_1 * Ig_slope
            ax.plot([d1,d2], [Ig1, Ig2],
                    linestyle='-',
                    color=line_col)
            ax.plot([di], [Ig_mid], 'ko',
                    markersize=mk_size)
            if k == n_rows - 1:
                ax.plot([d1,d2], [Ig1, Ig2],
                        linestyle='-',
                        color=line_col,
                        label='Ig'+Ig)

        ax.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b-%y"))
        ax.set_ylabel('OD')
        plt.legend(loc='best')
        plt.xticks(rotation=45)
        folder1= 'Braeden_oct_20_2022'
        folder2= 'oct_31_2022'
        max_n_inf_s = str(max_n_inf)
        fpure = 'slope_' + Ig + '_max_n_inf_' + max_n_inf_s + '.png'
        fname = os.path.join(self.parent.requests_path,
                folder1,
                folder2,
                fpure)
        fig.savefig(fname)

        plt.close('all')
        min_ratio = df[Ig_ratio_L].min()
        max_ratio = df[Ig_ratio_L].max()
        fig = pxp.histogram(df, x=Ig_ratio_L,
                color_discrete_sequence=Ig_color[Ig])
        fig.update_traces(xbins=dict( # bins used for histogram
                    start=min_ratio, end=max_ratio))
        fig.update_layout(font_size=20)
        fig.update_layout(hoverlabel={'font_size':20})
        fpure = 'ratio_' + Ig + '_max_n_inf_' + max_n_inf_s + '.html'
        fname = os.path.join(self.parent.requests_path,
                folder1,
                folder2,
                fpure)
        fig.write_html(fname)

        plt.close('all')
        min_slope = df[Ig_slope_L].min()
        max_slope = df[Ig_slope_L].max()
        fig = pxp.histogram(df, x=Ig_slope_L,
                color_discrete_sequence=Ig_color[Ig])
        fig.update_traces(xbins=dict( # bins used for histogram
                    start=min_slope, end=max_slope))
        fig.update_layout(font_size=20)
        fig.update_layout(hoverlabel={'font_size':20})
        fpure = 'slope_' + Ig + '_max_n_inf_' + max_n_inf_s + '.html'
        fname = os.path.join(self.parent.requests_path,
                folder1,
                folder2,
                fpure)
        fig.write_html(fname)


    def ahmads_request_dec_16_2022(self):

        use_manual_seaborn = True
        use_boxplot = False
        use_seaborn = False
        use_bars    = False

        fname  = 'lucas_data.xlsx'
        folder = 'Ahmad_dec_16_2022'
        fname = os.path.join('..','requests',folder, fname)
        df_up = pd.read_excel(fname)
        print(df_up)

        #Abbreviations
        dsd = 'Days since dose'
        dos = 'Dose of Sample'
        istat = 'Infection Status'
        binf = 'Binary Infection'

        #Relevant columns
        bio_columns = ['Wuhan',
                       'Beta',
                       'Omicron',
                       'Spike-IgG-5000',
                       'RBD-IgG-400',
                       'Spike-IgA-1000',
                       'RBD-IgA-100',
                      ]

        bio_labels = [
                'MNT50 Wuhan (SB.3)',
                'MNT50 Beta (B.1.351)',
                'MNT50 Omicron (BA.1)',
                'Spike IgG 1:5000',
                'RBD IgG 1:400',
                'Spike IgA 1:1000',
                'RBD IgA 1:100',
                ]

        bio_legend_xy = [
                'upper left',
                'upper left',
                'upper left',
                'upper left',
                'upper left',
                'upper left',
                'upper left',
                ]

        bio_legend_for_dose = [
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            ]

        bio_column_to_label       = {}
        bio_column_to_legend_xy   = {}
        bio_column_to_legend_for_dose = {}

        for x,y,z,w in zip(bio_columns,
                       bio_labels,
                       bio_legend_xy,
                       bio_legend_for_dose):
            bio_column_to_label[x]       = y
            bio_column_to_legend_xy[x]   = z
            bio_column_to_legend_for_dose[x] = w

        plots_that_use_ylog = ['Wuhan', 'Beta', 'Omicron']

        #Remove Dose #1
        selection = df_up[dos] == 1
        df_up.drop(selection[selection].index, inplace=True)
        n_doses = len(df_up[dos].drop_duplicates())

        #Map infection status
        dc = {'Not infected':'No Inf',
              'Infected in past 3 months':'Inf < 3mo',
              'Infected >3 months':'Inf > 3mo'}

        df_up[istat].replace('Not infected',
                             dc['Not infected'],
                              inplace=True)
        df_up[istat].replace('Infected in past 3 months',
                             dc['Infected in past 3 months'],
                              inplace=True)
        df_up[istat].replace('Infected >3 months',
                             dc['Infected >3 months'],
                              inplace=True)

        plot_istat_order = [
            dc['Not infected'],
            dc['Infected >3 months'],
            dc['Infected in past 3 months'],
                             ]
        #istat_to_row = {'No Inf':2, 'Inf > 3mo':1, 'Inf < 3mo':0}

        #Compute time labels
        max_days_since_dose = df_up[dsd].max()
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
        bins = pd.cut(df_up[dsd], intervals, labels=time_labels)

        if use_manual_seaborn:
            #Include column with 
            md_props = dict(linestyle='-',
                           linewidth=2,
                           color='black')
            bx_props = dict(linestyle='-',
                           linewidth=1,
                           alpha=0.95,
                           )
            wh_props = dict(linestyle='-',
                           linewidth=1,
                           color='black')
            cp_props = dict(linestyle='-',
                           linewidth=1,
                           color='black')
            my_pal = {'Inf Neg': 'blue', 'Inf Pos': 'red'}
            bins = pd.cut(df_up[dsd], intervals, labels=time_labels)
            df_up['mpd'] = bins

            selection = df_up['mpd']!='0- 2'
            selection = df_up['mpd']!='0- 2'
            df_up.drop(selection[selection].index, inplace=True)

            df_g = df_up.groupby(dos)
            #groups = [2,3,4,5]
            groups = [4,5]
            map_group_to_width = {4:0.8, 5:0.8}
            n_groups = len(groups)
            for bio_column in bio_columns:
                fig, ax = plt.subplots(ncols=n_groups,
                        figsize=(10, 5),
                        #gridspec_kw={'width_ratios': [5, 1]},
                        sharey=True)
                legend_for_dose = bio_column_to_legend_for_dose[bio_column]
                for k, group in enumerate(groups):
                    df_d = df_g.get_group(group)
                    sns.boxplot(ax = ax[k],
                            x   = 'mpd',
                            y   = bio_column,
                            hue = binf,
                            data=df_d,
                            showfliers=False,
                            #width=map_group_to_width[group],
                            medianprops  = md_props,
                            boxprops     = bx_props,
                            whiskerprops = wh_props,
                            capprops     = cp_props,
                            hue_order    = ['Inf -', 'Inf +'],
                            order        = ['0- 2'],
                            palette      = my_pal,
                            )
                    if group == legend_for_dose:
                        pos = bio_column_to_legend_xy[bio_column]
                        ax[k].legend(loc=pos)
                    else:
                        ax[k].legend([],[], frameon=False)
                    ax[k].set_ylabel('')
                    xlabel = 'Months after dose ' + str(group)
                    ax[k].set_xlabel(xlabel)
                    ax[k].tick_params(axis='x', rotation=90)
                    if bio_column in plots_that_use_ylog:
                        ax[k].set_yscale('log', base=2)
                        R = 2**np.arange(2,11,2,dtype=int)
                        ax[k].set_yticks(R)
                        y_labels = [str(x) for x in R]
                        ax[k].set_yticklabels(y_labels)
                    else:
                        ax[k].set_yticks([0,1,2,3])
                y_label = bio_column_to_label[bio_column]
                fig.supylabel(y_label, weight='bold')
                fname  = 'combo_' + bio_column + '.png'
                folder = 'Ahmad_dec_16_2022'
                fname = os.path.join('..','requests',folder, fname)
                plt.tight_layout()
                plt.savefig(fname)

        if use_seaborn:
            #Include column with 
            md_props = dict(linestyle='-',
                           linewidth=2,
                           color='black')
            bx_props = dict(linestyle='-',
                           linewidth=1,
                           alpha=0.95,
                           )
            wh_props = dict(linestyle='-',
                           linewidth=1,
                           color='black')
            cp_props = dict(linestyle='-',
                           linewidth=1,
                           color='black')
            my_pal = {'No Inf': 'blue', 'Inf > 3mo': 'orange', 'Inf < 3mo':'red'}
            df_up['mpd'] = bins
            df_g = df_up.groupby(dos)
            #groups = [2,3,4,5]
            groups = [4,5]
            map_group_to_width = {4:0.8, 5:0.8}
            n_groups = len(groups)
            for bio_column in bio_columns:
                fig, ax = plt.subplots(ncols=n_groups,
                        figsize=(10, 5),
                        gridspec_kw={'width_ratios': [5, 1]},
                        sharey=True)
                legend_for_dose = bio_column_to_legend_for_dose[bio_column]
                for k, group in enumerate(groups):
                    df_d = df_g.get_group(group)
                    sns.boxplot(ax = ax[k],
                            x='mpd',
                            y=bio_column,
                            data=df_d,
                            showfliers=False,
                            width=map_group_to_width[group],
                            medianprops  = md_props,
                            boxprops     = bx_props,
                            whiskerprops = wh_props,
                            capprops     = cp_props,
                            hue_order    = plot_istat_order,
                            palette      = my_pal,
                            hue=istat)
                    if group == legend_for_dose:
                        pos = bio_column_to_legend_xy[bio_column]
                        ax[k].legend(loc=pos)
                    else:
                        ax[k].legend([],[], frameon=False)
                    if group == 5:
                        ax[k].set_xlim(-0.5, 0.5)
                    else:
                        xp = 0.5
                        for i in range(n_time_labels-1):
                            ax[k].axvline(xp,color='gray')
                            xp += 1.0
                    ax[k].set_ylabel('')
                    xlabel = 'Months after dose ' + str(group)
                    ax[k].set_xlabel(xlabel)
                    ax[k].tick_params(axis='x', rotation=90)
                    if bio_column in plots_that_use_ylog:
                        ax[k].set_yscale('log', base=2)
                        R = 2**np.arange(2,11,2,dtype=int)
                        ax[k].set_yticks(R)
                        y_labels = [str(x) for x in R]
                        ax[k].set_yticklabels(y_labels)
                    else:
                        ax[k].set_yticks([0,1,2,3])
                y_label = bio_column_to_label[bio_column]
                fig.supylabel(y_label, weight='bold')
                fname  = 'combo_' + bio_column + '.png'
                folder = 'Ahmad_dec_16_2022'
                fname = os.path.join('..','requests',folder, fname)
                plt.tight_layout()
                plt.savefig(fname)

        if use_boxplot:

            #Include column with 
            df_up['mpd'] = bins
            color_vec = ['lightblue','orange','lightgreen']
            #Group df
            df_g = df_up.groupby(istat)
            for bio_column in bio_columns:
                fig, ax = plt.subplots(nrows=3, sharex=True)
                ymax = 0
                ymin = 0
                for ci, i_status in enumerate(plot_istat_order):
                    md_props = dict(linestyle='-',
                                   linewidth=3,
                                   color='black')
                    bx_props = dict(linestyle='-',
                                   linewidth=1,
                                   color='black')
                    wh_props = dict(linestyle='-',
                                   linewidth=1,
                                   color='black')
                    #Infection status
                    df_i = df_g.get_group(i_status)
                    bp = df_i.boxplot(ax = ax[ci],
                                 column=[bio_column],
                                 by=[dos, 'mpd'],
                                 grid=False,
                                 patch_artist=True,
                                 rot = 90,
                                 medianprops = md_props,
                                 boxprops = bx_props,
                                 whiskerprops = wh_props,
                                 showfliers = False,
                                 return_type = 'dict',
                                 )
                    #print(df_i)
                    #print(bp[bio_column].keys())
                    #return
                    for patch in bp[bio_column]['boxes']:
                        patch.set_facecolor(color_vec[ci])
                    ymax = np.maximum(ax[ci].get_ylim()[1], ymax)
                    xp = 1
                    for k in range(n_doses-1):
                        xp += n_time_labels
                        ax[ci].axvline(xp-0.5,color='k')
                for ci, i_status in enumerate(plot_istat_order):
                    if bio_column in plots_that_use_ylog:
                        #ax[ci].set_ylim([1,ymax])
                        ax[ci].set_yscale('log', base=2)
                        R = 2**np.arange(2,11,2,dtype=int)
                        ax[ci].set_yticks(R)
                        y_labels = [str(x) for x in R]
                        ax[ci].set_yticklabels(y_labels)
                    else:
                        #ax[ci].set_ylim([0,ymax])
                        ax[ci].set_yticks([0,1,2,3])
                    ax[ci].set_ylabel(i_status)
                    ax[ci].set_xlabel('')
                    ax[ci].set_title('')
                ax[ci].set_xlabel('(Dose #, Months after dose)')
                plt.xticks(rotation=90)
                plt.suptitle('')
                plt.title('')
                y_label = bio_column_to_label[bio_column]
                fig.supylabel(y_label, weight='bold')
                fname  = 'box_' + bio_column + '.png'
                folder = 'Ahmad_dec_16_2022'
                fname = os.path.join('..','requests',folder, fname)
                plt.tight_layout()
                plt.savefig(fname)


        if use_bars:

            df_g = df_up.groupby([df_up[dos],
                bins, df_up[istat]])

            generate_descriptive_stats = False
            if generate_descriptive_stats:
                descriptor = df_g.describe()[bio_columns]
                fname  = 'Ahmad_request_20_12_2022.xlsx'
                folder = 'Ahmad_dec_16_2022'
                fname = os.path.join('..','requests',folder, fname)
                descriptor.to_excel(fname)

            df_g = df_g.agg('median', numeric_only=True)

            for bio_column in bio_columns:
                #Move the Infection Status as a header
                df_u = df_g[bio_column].unstack(level=2)

                #Remove all except the first time label for
                #dose #5
                selection = df_u.loc[(5,slice('2- 4','8-10')),:]
                df_u.drop(selection.index, inplace=True)
                print(df_u)

                #Time to plot
                fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
                ymax = 0
                color_vec = ['blue','orange','red']
                for ci, i_status in enumerate(plot_istat_order):
                    df_u[i_status].plot(ax=ax[ci],
                                      kind='bar',
                                      color=color_vec[ci])
                    xp = 0
                    for k in range(n_doses-1):
                        xp += n_time_labels
                        ax[ci].axvline(xp-0.5,color='k')
                    ax[ci].set_xlabel('(Dose #, Months after dose)')
                    ax[ci].set_ylabel(column)
                    ymax = np.maximum(ax[ci].get_ylim()[1], ymax)
                for ci, column in enumerate(df_u.columns):
                    ax[ci].set_ylim([0,ymax])
                fname  = bio_column + '.png'
                folder = 'Ahmad_dec_16_2022'
                fname = os.path.join('..','requests',folder, fname)
                plt.tight_layout()
                plt.savefig(fname)

    def boxplots_using_L_file(self):

        fname  = 'L_avec_metadata.xlsx'
        folder = 'Jessica_jan_23_2023'
        fname = os.path.join('..','requests',folder, fname)
        df_up = pd.read_excel(fname)
        site_selection = df_up['Site'].isin([20,61])
        #print(len(df_up))
        df_up = df_up[~site_selection]
        #print(len(df_up))
        print(df_up)

        use_swarm = True
        Z2T = '0-2'
        T2F = '2-4'


        #Abbreviations
        i_negative = 'Not Infected'
        i_positive = 'Infected'

        MSD = 'Months since dose'
        #dsd = 'Days since dose'

        NPCD= 'Nearest pre-collection dose'
        #dos = 'Dose of Sample'

        BIS = 'Binary infection status'
        #binf = 'Binary Infection'

        #Relevant columns
        bio_columns = ['Wuhan (SB3)',
                       'Beta (B.1.351)',
                       'Omicron (BA.1)',
                       'Spike-IgG-5000',
                       'RBD-IgG-400',
                       'Spike-IgA-1000',
                       'RBD-IgA-100',
                      ]

        bio_thresholds = [10, 10, 10, 0.0999, 0.4389, 0.1788, 0.5637]

        bio_labels = [
                'MNT50 Wuhan (SB.3)',
                'MNT50 Beta (B.1.351)',
                'MNT50 Omicron (BA.1)',
                'Spike IgG (AU)',
                'RBD IgG (AU)',
                'Spike IgA (AU)',
                'RBD IgA (AU)',
                ]

        bio_column_to_label       = {}
        bio_column_to_threshold   = {}

        for x,y,z in zip(bio_columns, bio_labels, bio_thresholds):
            bio_column_to_label[x]       = y
            bio_column_to_threshold[x]   = z

        plots_that_use_ylog = ['Wuhan (SB3)',
                               'Beta (B.1.351)',
                               'Omicron (BA.1)',
                              ]

        #Remove Dose #1
        #selection = df_up[dos] == 1
        #df_up.drop(selection[selection].index, inplace=True)
        #n_doses = len(df_up[dos].drop_duplicates())

        md_props = dict(linestyle='-',
                       linewidth=2,
                       color='black')
        bx_props = dict(linestyle='-',
                       linewidth=1,
                       alpha=0.95,
                       )
        wh_props = dict(linestyle='-',
                       linewidth=1,
                       color='black')
        cp_props = dict(linestyle='-',
                       linewidth=1,
                       color='black')

        #Specific format due to the elimination of labels after 2-4
        df_up[MSD] = df_up[MSD].str.replace('0- 2', Z2T)
        df_up[MSD] = df_up[MSD].str.replace('2- 4', T2F)

        selection  = df_up[MSD]== Z2T
        selection |= df_up[MSD]== T2F
        selection  = ~selection
        df_up.drop(selection[selection].index, inplace=True)

        df_g = df_up.groupby(NPCD)
        groups = [4,5]
        map_group_to_width = {4:0.8, 5:0.8}
        n_groups = len(groups)


        generate_descriptive_stats = True
        if generate_descriptive_stats:

            selection  = df_up[NPCD] == 4
            selection |= df_up[NPCD] == 5
            selection  = ~selection
            df_up.drop(selection[selection].index, inplace=True)

            df_g = df_up.groupby([df_up[NPCD], df_up[MSD], df_up[BIS]])
            descriptor = df_g.describe()[bio_columns]
            fname  = 'stats_L_x.xlsx'
            folder = 'Jessica_jan_23_2023'
            fname = os.path.join('..','requests',folder, fname)
            descriptor.to_excel(fname)
            return

        #Color palette for boxplots
        my_bp_palette = {i_negative: 'blue', i_positive: 'red'}

        #Color palette for swarm plots
        my_sw_palette = {i_negative: 'black', i_positive: 'black'}

        for bio_column in bio_columns:
            print(f'{bio_column=}')
            if bio_column == 'Beta (B.1.351)':
                #This plot causes a key error due to empty data.
                continue
            y_val = bio_column_to_threshold[bio_column]
            fig, ax = plt.subplots(ncols=n_groups, sharey=True)
            for k, group in enumerate(groups):
                df_d = df_g.get_group(group)
                sns.boxplot(ax = ax[k],
                            x  = MSD,
                            y   = bio_column,
                            hue = BIS,
                            data=df_d,
                            showfliers=False,
                            medianprops  = md_props,
                            boxprops     = bx_props,
                            whiskerprops = wh_props,
                            capprops     = cp_props,
                            hue_order    = [i_negative, i_positive],
                            order        = [Z2T, T2F],
                            palette      = my_bp_palette,
                           )
                sns.stripplot(x         = MSD,
                              y         = bio_column,
                              hue       = BIS,
                              data      = df_d,
                              dodge     = True,
                              hue_order = [i_negative, i_positive],
                              order     = [Z2T, T2F],
                              palette   = my_sw_palette,
                              size      = 2,
                              legend    = False,
                              ax = ax[k])
                ax[k].axhline(y_val,color='gray', label='Threshold')
                ax[k].legend([],[], frameon=False)
                ax[k].set_ylabel('')
                xlabel = 'Months after dose ' + str(group)
                ax[k].set_xlabel(xlabel)
                ax[k].tick_params(axis='x', rotation=90)
                if bio_column in plots_that_use_ylog:
                    ax[k].set_yscale('log', base=2)
                    R = 2**np.arange(2,11,2,dtype=int)
                    ax[k].set_yticks(R)
                    y_labels = [str(x) for x in R]
                    ax[k].set_yticklabels(y_labels)
                else:
                    ax[k].set_yticks([0,1,2,3])
            y_label = bio_column_to_label[bio_column]
            fig.supylabel(y_label, weight='bold')
            if use_swarm:
                fname  = 'swarm_' + bio_column + '.png'
            else:
                fname  = 'bplot_' + bio_column + '.png'
            folder = 'Jessica_jan_23_2023'
            fname = os.path.join('..','requests',folder, fname)
            print(f'{fname=}')
            plt.legend(bbox_to_anchor=(1.4, 0.45), loc='center', borderaxespad=0)
            plt.tight_layout()
            plt.savefig(fname)
            plt.close('all')

    def generate_plot_for_time_between_infection_and_death(self):
        #Time between infection and death.
        folder = 'Tara_jan_05_2023'
        fname = 'Time_between_infection_and_death.xlsx'
        fname = os.path.join(self.requests_path, folder, fname)
        df = pd.read_excel(fname)
        print(df)
        WBIAD = 'Weeks between infection and death'
        sns.histplot(data=df, x = WBIAD, discrete=False, binwidth=4)
        #sns.histplot(data=df, x = WBIAD, discrete=True)

        folder = 'Tara_jan_05_2023'
        fname = 'plot_time_between_infection_and_death.png'
        fname = os.path.join(self.requests_path, folder, fname)
        plt.savefig(fname)
        plt.close('all')

    def generate_report_for_time_between_infection_and_death(self):
        #Time between infection and death.
        self.parent.LIS_obj.add_n_infections_column()
        WBIAD = 'Weeks between infection and death'
        DBIAD = 'Days between infection and death'
        NIBD  = 'Nearest infection before death'
        DOI   = 'Date of infection'
        DOR   = self.parent.MPD_obj.DOR
        list_of_columns = ['ID',
                           DOR,
                           'Reason',
                           '# infections',
                           NIBD,
                           DOI,
                           DBIAD,
                           WBIAD]

        list_of_columns += self.parent.LIS_obj.positive_date_cols
        list_of_columns += self.parent.LIS_obj.positive_type_cols
        list_of_columns += self.parent.LIS_obj.wave_of_inf_cols
        list_of_columns += self.parent.LIS_obj.vaccine_type_cols
        list_of_columns += self.parent.LIS_obj.vaccine_date_cols

        self.parent.df[DBIAD] = np.nan
        self.parent.df[NIBD]  = np.nan
        self.parent.df[DOI]   = np.nan
        L = []
        for index, row in self.parent.df.iterrows():

            reason = row['Reason']
            if reason != 'Deceased':
                continue

            dor = row[self.parent.MPD_obj.DOR]
            if pd.isnull(dor):
                continue

            n_inf = row['# infections']
            if n_inf == 0:
                continue

            inf_dates = pd.to_datetime(row[self.parent.LIS_obj.positive_date_cols])
            deltas = (dor - inf_dates) / np.timedelta64(1,'D')
            selection = deltas < 0
            deltas = deltas[~selection]
            if len(deltas) == 0:
                continue
            L.append(index)
            deltas = deltas.sort_values()
            days_between_inf_and_death = deltas.iloc[0]
            weeks_between_inf_and_death = days_between_inf_and_death / 7.
            weeks_between_inf_and_death = np.ceil(weeks_between_inf_and_death)
            if weeks_between_inf_and_death < 1:
                #weeks_between_inf_and_death  += 1
                pass
            nearest_infection = deltas.index[0]
            doi = row[nearest_infection]
            self.parent.df.loc[index, DBIAD] = days_between_inf_and_death
            self.parent.df.loc[index, WBIAD] = weeks_between_inf_and_death
            self.parent.df.loc[index, NIBD]  = nearest_infection
            self.parent.df.loc[index, DOI]   = doi
        df_w = self.parent.df.loc[L,:].copy()
        df_w = df_w[list_of_columns]
        folder = 'Tara_jan_05_2023'
        fname = 'Time_between_infection_and_death.xlsx'
        fname = os.path.join(self.requests_path, folder, fname)
        df_w.to_excel(fname, index=False)

    def track_serology_with_infections(self):
        #Use the periods between dates of collection
        #to identify trends between participants.
        #folder = 'Tara_jan_05_2023'
        fname = 'W.xlsx'
        fname = os.path.join(self.outputs_path, fname)
        for index_w, row_w in df_w.iterrows():
