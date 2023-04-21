#JRR @ McMaster University
#Update: 10-Oct-2022
import numpy as np
import pandas as pd
#pd.options.plotting.backend = 'plotly'
#import plotly.express as pxp
import os
import csv
import re
#import PIL
from tqdm import tqdm as pbar
#import networkx as nx
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
#mpl.rcParams['font.family'] = 'monospace'
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter as KMFitter
from lifelines.statistics import logrank_test as LRT
from scipy.stats import mannwhitneyu as MannWhitney
from scipy.stats import chi2_contingency as chi_sq_test
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
        self.outputs_path = os.path.join('..', 'outputs')

        #Cox Proportional Hazard Model
        self.cph = None


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

        #fname = 'W.xlsx'
        #fname = os.path.join(self.outputs_path, fname)
        #df_w  = pd.read_excel(fname)

        #Id to sample
        ID_to_samples = {}

        #Id to delta infection 
        #Days post dose
        ID_to_delta_inf = {}
        ID_to_delta_only_lower_inf = {}
        ID_to_delta_within_inf = {}

        vaccine_index = 4

        ref_a = 3*30
        ref_b = 6*30
        ref_days_post  = [ref_a, ref_b]

        tolerance_between_ref_and_days_post_dose = 30 #days
        tolerance_between_inf_and_ref_a = 3*30 #days
        ref_a_minus_tol = ref_a - tolerance_between_inf_and_ref_a 

        DOC = 'Date Collected'
        vaccine_dates_h = self.parent.LIS_obj.vaccine_date_cols
        inf_dates_h = self.parent.LIS_obj.positive_date_cols
        for index_m, row_m in self.parent.df.iterrows():

            ID = row_m['ID']

            vaccine_dates = row_m[vaccine_dates_h]
            vaccine_date = vaccine_dates.iloc[vaccine_index-1]

            if pd.isnull(vaccine_date):
                continue

            lower_bound = vaccine_date + pd.DateOffset(days=ref_a_minus_tol)
            ref_date_a = vaccine_date + pd.DateOffset(days=ref_a)
            ref_date_b = vaccine_date + pd.DateOffset(days=ref_b)
            ref_dates = [ref_date_a, ref_date_b]

            selector = self.parent.LSM_obj.df['ID'] == ID
            if not selector.any():
                continue

            df_lsm = self.parent.LSM_obj.df[selector]

            flag_vector    = np.full(2,False)
            full_ID_vector = [0,0]
            delta_vector   = [0,0]
            doc_vector     = [0,0]

            for index_s, row_s in df_lsm.iterrows():
                full_ID = row_s['Full ID']
                doc = row_s[DOC]

                if doc <= vaccine_date:
                    #We need it to be posterior to the vaccination date
                    continue


                for k, ref_date in enumerate(ref_dates):

                    delta = (doc - ref_date) / np.timedelta64(1,'D')
                    dist  = np.abs(delta)

                    if dist < tolerance_between_ref_and_days_post_dose:
                        flag_vector[k] = True
                        delta_vector[k] = delta
                        full_ID_vector[k] = full_ID
                        doc_vector[k] = doc

            if flag_vector.all():

                ID_to_samples[ID] = [full_ID_vector, delta_vector, doc_vector]

                inf_dates = row_m[inf_dates_h]


                if inf_dates.count() == 0:
                    #Next candidate
                    continue


                selector = inf_dates.notnull()
                inf_dates = inf_dates[selector]

                #Infection happened before or 
                #at the RIGHT date of collection.
                selector = inf_dates  <= doc_vector[1]

                if not selector.any():
                    #Next candidate
                    continue

                inf_dates = inf_dates[selector]

                #After or at the LEFT reference point 
                #minus the tolerance.
                #In this case, the vaccination date.
                selector = lower_bound <= inf_dates

                if not selector.any():
                    #Next candidate
                    continue

                inf_dates = pd.to_datetime(inf_dates[selector])
                delta_inf = (inf_dates - vaccine_date) / np.timedelta64(1,'D')

                ID_to_delta_inf[ID] = delta_inf

                if 1 < delta_inf.count():
                    #print(f'{ID=} has {delta_inf.count()} infections')
                    pass

                #Infection is at or after the LEFT date of collection.
                selector = doc_vector[0] <= inf_dates

                if not selector.any():
                    #Next candidate
                    continue

                inf_dates = inf_dates[selector]

                ID_to_delta_within_inf[ID] = inf_dates

        bio_params = ['Nuc-IgG-100', 'Nuc-IgA-100']
        folder = self.parent.LSM_path
        fname = 'serology_thresholds.xlsx'
        fname = os.path.join(folder, fname)
        df_t = pd.read_excel(fname)

        ticks = np.array([0,30,60,90,120,150,180,210], dtype=int)
        labels = ticks // 30

        for bio_param in bio_params:

            print(f'{bio_param=}')

            folder_1 = 'Ahmad_feb_06_2023'
            folder_2 = bio_param
            folder = os.path.join(self.requests_path, folder_1, folder_2)
            if os.path.exists(folder):
                pass
            else:
                os.makedirs(folder)

            selector = df_t['Ig'] == bio_param
            if not selector.any():
                print(f'{bio_param=}')
                raise ValueError('Bio parameter not found')
            bio_t = df_t.loc[selector, 'Threshold'].iloc[0]
            fig_g, ax_g = plt.subplots()
            counter = 0

            n_cases = 0
            n_inf_within = 0
            n_inf = 0
            n_only_lower_inf = 0
            #================Y MAX==================
            y_max = 0
            for ID, V in ID_to_samples.items():
                full_ID_vector = V[0]
                Y = []
                for full_ID in full_ID_vector:
                    s = self.parent.LSM_obj.df['Full ID'] == full_ID
                    row_s = self.parent.LSM_obj.df[s].iloc[0]
                    bio_value = row_s[bio_param]
                    Y.append(bio_value)
                if Y[0] < bio_t and bio_t < Y[1]:
                    n_cases += 1
                    if y_max < Y[1]:
                        y_max = Y[1]
            #================END Y MAX==================
            for ID, V in ID_to_samples.items():
                full_ID_vector = V[0]
                delta_vector   = V[1]
                X = []
                Y = []
                for full_ID, delta, ref in zip(full_ID_vector,
                        delta_vector, ref_days_post):
                    s = self.parent.LSM_obj.df['Full ID'] == full_ID
                    row_s = self.parent.LSM_obj.df[s].iloc[0]
                    bio_value = row_s[bio_param]
                    Y.append(bio_value)
                    X.append(ref + delta)
                if Y[0] < bio_t and bio_t < Y[1]:
                    counter += 1
                    bio_delta = Y[1] - Y[0]
                    bio_delta_str = '{:0.2f}'.format(bio_delta)
                    if ID in ID_to_delta_inf:
                        n_inf += 1
                        if ID in ID_to_delta_within_inf:
                            n_inf_within += 1
                            ax_g.plot(X,Y,'r-',marker = 'o')
                        else:
                            ax_g.plot(X,Y,'r--',marker = 'o')
                    else:
                        ax_g.plot(X,Y,'b-',marker = 'o')
                    fig, ax = plt.subplots()
                    ax.plot(X,Y,'b-',marker = 'o')
                    ax.axvline(x = ref_a, color='k', linewidth=2)
                    ax.axvline(x = ref_b, color='k', linewidth=2)
                    ax.axhline(y = bio_t, color='gray', linewidth=2)
                    ax.set_xticks(ticks)
                    ax.set_xticklabels(labels)
                    ax.set_xlabel('Months post-4th dose')
                    ax.set_ylabel(bio_param)
                    ax.set_ylim([0,y_max])
                    ax.set_title(ID + ': $\Delta$=' + bio_delta_str)
                    if ID in ID_to_delta_inf:
                        cax = ax.twinx()
                        cax.get_yaxis().set_visible(False)
                        delta_inf_vector = ID_to_delta_inf[ID]
                        for delta_inf in delta_inf_vector:
                            cax.plot(delta_inf,0.5,'ro')
                    fname = str(counter) + '.png'
                    fname = os.path.join(folder, fname)
                    fig.savefig(fname)
                    plt.close(fig)

            n_only_lower_inf = n_inf - n_inf_within
            print(f'{n_cases=}')
            print(f'{n_inf=}')
            print(f'{n_inf_within=}')
            print(f'{n_only_lower_inf=}')
            print('==========================')

            ax_g.axvline(x = ref_a, color='k', linewidth=2)
            ax_g.axvline(x = ref_b, color='k', linewidth=2)
            ax_g.axhline(y = bio_t, color='gray', linewidth=2)
            ax_g.set_xticks(ticks)
            ax_g.set_xticklabels(labels)
            ax_g.set_xlabel('Months post-4th dose')
            ax_g.set_ylabel(bio_param)
            ax_g.set_ylim([0,y_max])
            fname = 'global.png'
            fname = os.path.join(folder, fname)
            fig_g.savefig(fname)
            plt.close('all')


    def plot_infections_on_bars(self):
        folder='Megan_feb_13_2023'
        fname = 'tri_merge.xlsx'
        fname = os.path.join(self.requests_path, folder, fname)
        df_i = pd.read_excel(fname, sheet_name = 'Infection_column')
        print(df_i)

        intervals = pd.date_range(start='2019-12-31', end='2023-02-01', freq='M')
        periods   = pd.period_range(start='2020-01', end='2023-01', freq='M')
        periods   = periods.to_timestamp().strftime('%b-%y')
        #print(intervals)
        #print(periods)
        DOI = 'Infection date'
        bins = pd.cut(df_i[DOI], intervals, labels=periods)
        grouped_dates = df_i[DOI].groupby([df_i['Site'], bins]).agg('count')
        #print(grouped_dates[(1,)])
        grouped_dates = grouped_dates.rename('Count')
        df = grouped_dates.reset_index()
        print(df)
        fig, ax = plt.subplots()
        sns.barplot(ax = ax, data=df, x='Infection date',y='Count',hue='Site')
        #grouped_dates.plot(ax=ax)

        plt.xticks(rotation=90)
        plt.tight_layout()

        fname = 'bars.png'
        fname = os.path.join(self.requests_path, folder, fname)
        fig.savefig(fname)

    def plot_infections_on_map(self):
        #Plot the infections on top of a map
        folder='Megan_feb_13_2023'
        fname = 'coordinates_with_deltas.xlsx'
        fname = os.path.join(self.requests_path, folder, fname)
        #Data frame with coordinates
        df_c = pd.read_excel(fname, dtype={'Site':np.int32,
            'ID':np.int32, 'Index':np.int32,
            'x':np.int32, 'y':np.int32,
            'x_L':np.int32, 'y_L':np.int32})
        site_to_ID = {}
        ID_to_label = {}
        visited_ID = {}
        for index, row in df_c.iterrows():
            ID = row['Index']
            if ID == 0:
                continue
            site = row['Site']
            name = row['Short Name']
            site_to_ID[site] = ID
            txt = str(ID) + name
            if ID <= 10:
                txt = '{:22}'.format(txt)
            else:
                txt = '{:19}'.format(txt)
            #print(len(txt))
            ID_to_label[ID] = txt
            visited_ID[ID] = False

        #print(df_c)
        fname = 'tri_merge.xlsx'
        fname = os.path.join(self.requests_path, folder, fname)
        #Data frame with infections
        df_i = pd.read_excel(fname, sheet_name = 'Infection_column')

        intervals = pd.date_range(start='2019-12-31', end='2023-02-01', freq='M')
        periods   = pd.period_range(start='2020-01', end='2023-01', freq='M')
        periods   = periods.to_timestamp().strftime('%b-%y')

        DOI = 'Infection date'
        bins = pd.cut(df_i[DOI], intervals, labels=periods)
        grouped_dates = df_i[DOI].groupby([df_i['Site'], bins]).agg('count')
        grouped_dates = grouped_dates.rename('Count')
        df_i = grouped_dates.reset_index().groupby('Infection date')
        fc = 10
        lw = 2
        for k, (g_name, df_g) in enumerate(df_i):
            if 100 < k:
                break
            print(f'Plotting for {g_name=}')
            fig, ax = plt.subplots()
            self.reuse_plot_infections_on_map(folder, df_c, ax)
            ID_to_label_m = ID_to_label.copy()
            visited_ID_m = visited_ID.copy()
            ltc_cases = 0
            rh_cases = 0
            for index_g, row_g in df_g.iterrows():

                site = row_g['Site']
                s = df_c['Site'] == site
                s_index = s[s].index[0]

                ID = df_c.loc[s_index,'Index']

                if ID == 0:
                    continue

                count = row_g['Count']
                if site < 30:
                    ltc_cases += count
                else:
                    rh_cases += count

                if visited_ID_m[ID]:
                    ID_to_label_m[ID] += ',' + str(count)
                else:
                    visited_ID_m[ID] = True
                    ID_to_label_m[ID] += str(count)


                if count == 0:
                    pass
                else:
                    print('Site:', site, ' Count:', count)
                    #ID = site_to_index[site]
                    x = df_c.loc[s_index, 'x']
                    y = df_c.loc[s_index, 'y']
                    orientation = df_c.loc[s_index,'Orientation']
                    color = df_c.loc[s_index,'Color']
                    style = color + '-'

                    if orientation == 'U':
                        #Vertical
                        dx = 0
                        dy = -count * fc
                    if orientation == 'D':
                        #Vertical
                        dx = 0
                        dy = count * fc
                    elif orientation == 'R':
                        #Horizontal
                        dy = 0
                        dx = count * fc


                    if site < 30:
                        x += 0
                    else:
                        x += 10

                    xn = x + dx
                    yn = y + dy

                    ax.plot([x,xn],[y,yn],style, linewidth=lw)

            #self.plot_names_on_map(df_c, ax, ID_to_label_m)
            y = 650
            x = 900
            ax.plot([x, x+125], [y, y], 'k-', linewidth=4)
            #txt = 'LTC(' + str(ltc_cases).zfill(2) + ')'
            txt = 'Long-Term-Care    (' + str(ltc_cases).zfill(2) + ')'
            ax.text(x+130, y, txt, fontsize=14, color='k')

            ax.plot([x, x+125], [y+100, y+100], 'b-', linewidth=4)
            txt = 'Retirement Home (' + str(rh_cases).zfill(2) + ')'
            ax.text(x+130, y+100, txt, fontsize=14, color='b')

            ax.text(1430, y-200, g_name, fontsize=18)
            ax.axis('off')
            #fname = g_name + '.png'
            #fname = str(k).zfill(2) + '.png'
            fname = 'im' + str(k).zfill(2) + '.png'
            fname = os.path.join(self.requests_path, folder, 'plots', fname)
            fig.savefig(fname, bbox_inches='tight', pad_inches=0)
            plt.close('all')





    def reuse_plot_infections_on_map(self, folder, df_c, ax):
        #Plot the infections on top of a map
        fname = 'map_v4.png'
        iname = os.path.join(self.requests_path, folder, fname)
        im = mpl.image.imread(iname)
        ax.imshow(im)

        for index, row in df_c.iterrows():
            if row['ID'] == 0:
                continue
            x = row['x']
            y = row['y']
            x_L = row['x_L']
            y_L = row['y_L']
            site = row['Site']
            ID = row['ID']
            c = row['Coordinates']
            ax.plot(x,y,'wo', markersize=3)
            x_new = x + x_L
            y_new = y + y_L
            #ax.text(x_new, y_new, ID)


        #Shalom Village line
        x = 550
        y = 1000
        dx = 55
        dy = -55
        #ax.plot([x,x+dx],[y, y+dy],'-', color='gray')

    def plot_names_on_map(self, df_c, ax, ID_to_label):
        rexp = re.compile(',(?P<number>[0-9]+)')

        for index, row in df_c.iterrows():
            ID   = row['ID']
            if ID == 0:
                continue
            txt = ID_to_label[ID]
            x = row['label_x']
            y = row['label_y']
            obj = rexp.search(txt)
            if obj:
                n = obj.group('number')
                sub = obj.group(0)
                start = txt.replace(sub,',')
                l_start = len(start)
                end = ' ' * l_start  + n
                ax.text(x, y, end, fontsize=8, color='b')
                ax.text(x, y, start, fontsize=8, color='k')
            else:
                if ID in [14,15,16]:
                    ax.text(x, y, txt, fontsize=8, color='b')
                else:
                    ax.text(x, y, txt, fontsize=8, color='k')


    def draw_inf_vac_history_from_serology_for_sheraton(self):
        #This function is used to plot the serology trajectory
        #of a given participant.

        #The original version of this function was created to
        #satisfy the request stipulated on the folder:
        #main_folder = 'Tara_feb_21_2023'

        main_folder = 'Sheraton_mar_23_2023'
        bio_list = ['Nuc-IgG-100']
        date_format = mpl.dates.DateFormatter('%b-%y')
        study_start_date = datetime.datetime(2022,7,1)
        study_end_date = datetime.datetime(2022,9,13)

        #The serology thresholds are stored in the
        #following folder.
        fname = 'serology_thresholds.xlsx'
        fname = os.path.join(self.parent.LSM_path, fname)
        df_t = pd.read_excel(fname)

        #++To the function for the Sheraton project.
        #List of those participants that had a double
        #omicron infection (before BA.5).
        fname = 'double_o.xlsx'
        folder= 'Sheraton_mar_23_2023'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df_do = pd.read_excel(fname)

        #Filter the Master data frame for the double o individuals.
        s = self.parent.df['ID'].isin(df_do['ID'])
        df_m = self.parent.df[s].copy()

        inf_date_h = self.parent.LIS_obj.positive_date_cols
        vac_date_h = self.parent.LIS_obj.vaccine_date_cols

        #The counter is simply used to 
        #manually limit the number
        #of iterations.
        counter = 0

        N = df_m.shape[0]
        for index_m, row_m in pbar(df_m.iterrows(), total=N):
            counter += 1
            #if 2 < counter:
                #break
            ID = row_m['ID']
            s = self.parent.LSM_obj.df['ID'] == ID
            if not s.any():
                continue
            df_s = self.parent.LSM_obj.df[s]

            for bio in bio_list:

                #This list will store the date of collection
                #and the value of the biological parameter.
                L = []

                s = df_t['Ig'] == bio
                threshold = df_t.loc[s,'Threshold'].iloc[0]
                #Create a list of the value of
                #the biological parameter and the 
                #date of collection.
                for index_s, row_s in df_s.iterrows():
                    bio_value = row_s[bio]
                    if pd.notnull(bio_value):
                        doc = row_s['Date Collected']
                        L.append((doc, bio_value))
                #At least two points
                if 1 < len(L):
                    fig, ax = plt.subplots()

                    #Create DF.
                    df = pd.DataFrame(L, columns=['Time',bio])

                    #Was the threshold crossed?
                    flag_crossed_threhold = False
                    if df[bio].min() < threshold and threshold < df[bio].max():
                        flag_crossed_threhold = True
                    else:
                        #Did not cross.
                        flag_below = False
                        if df[bio].min() < threshold:
                            flag_below = True

                    #Serology time
                    df.plot(ax=ax, x='Time', y=bio, kind='line', marker='o', color='blue')

                    #Infection time
                    flag_had_infections = False
                    inf_dates = row_m[inf_date_h]

                    if 0 < inf_dates.count():
                        flag_had_infections = True
                        s = inf_dates.notnull()
                        inf_dates = inf_dates[s]

                        flag_infection_within = False

                        #Iterate over infection times.
                        for _ , inf_date in inf_dates.items():
                            ax.axvline(inf_date, color='red', linewidth=3)
                            #If there was an infection between the 
                            #earliest time of collection and the
                            #latest time of collection, then we had
                            #an infection within.
                            c1 = df['Time'].min() <= inf_date
                            c2 = inf_date <= df['Time'].max()
                            if c1 and c2:
                                flag_infection_within = True

                    #Vaccination time
                    vac_dates = row_m[vac_date_h]
                    if 0 < vac_dates.count():
                        s = vac_dates.notnull()
                        vac_dates = vac_dates[s]
                        for _ , date in vac_dates.items():
                            ax.axvline(date, color='black', linestyle='--', linewidth=2)

                    #Plot threshold line
                    ax.axhline(threshold, color='gray', linewidth=3)

                    #Plot study start and end date lines
                    ax.axvline(study_start_date, color='brown',
                               linestyle='dashdot', linewidth=3)
                    ax.axvline(study_end_date, color='brown',
                               linestyle='dashdot', linewidth=3)

                    ax.xaxis.set_major_formatter(date_format)
                    ax.set_title(ID)
                    ax.set_ylabel(bio)
                    ax.get_legend().remove()
                    fname = ID + '.png'

                    if flag_had_infections:
                        inf_folder = 'infected'
                        if flag_infection_within:
                            inf_folder = os.path.join(inf_folder, 'inside')
                        else:
                            inf_folder = os.path.join(inf_folder, 'outside')
                    else:
                        inf_folder = 'not_infected'

                    if flag_crossed_threhold:
                        crossed_folder = 'crossed'
                    else:
                        crossed_folder = 'not_crossed'
                        if flag_below:
                            crossed_folder = os.path.join(crossed_folder, 'below')
                        else:
                            crossed_folder = os.path.join(crossed_folder, 'above')

                    storage_folder = os.path.join(self.requests_path,
                            main_folder,
                            bio,
                            inf_folder,
                            crossed_folder)
                    if not os.path.exists(storage_folder):
                        os.makedirs(storage_folder)
                    fname = os.path.join(storage_folder, fname)
                    fig.savefig(fname, bbox_inches='tight', pad_inches=0)
                    plt.close('all')


    def dawns_request_feb_24_2023(self):
        data = '''Australia Israel Norway Slovenia Hungary Austria Germany Portugal Canada
        OECD(mean) U.S. Ireland Netherlands France Italy U.K. Spain Belgium'''
        values = '''28 58 58 9 7 19 34 25 81 38 31 56 15 48 32 27 66 50'''
        rexp = re.compile('[0-9a-zA-Z.)(]+')
        countries = rexp.findall(data)
        values = rexp.findall(values)
        values = [int(x) for x in values]
        df = pd.DataFrame({'Country':countries, 'Percentage':values})
        df = df.sort_values('Percentage', ascending=False)
        s = df['Country'].isin(['Canada', 'OECD(mean)'])
        df1 = df[s].copy()
        df2 = df[~s].copy()
        df3 = pd.concat((df1, df2), join='outer')
        print(df1)
        print(df2)
        print(df3)
        fig, ax = plt.subplots()
        colors = ['red' if k == 0 else 'blue' for k in range(len(df))]
        sns.barplot(ax = ax,
                x='Percentage',
                y='Country',
                data = df3,
                label='LTC',
                palette=colors)
        ax.bar_label(ax.containers[0])
        folder = 'Dawn_feb_24_2023'
        fname = 'plot.png'
        fname = os.path.join(self.requests_path, folder, fname)
        ax.set_ylabel('')
        fig.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close('all')

    def generate_poster_data_sheraton(self):
        #Mar 23 2023
        #Generate the dataset to replicate Ahmad's 
        #results for the poster.
        L = []
        multi_omicron = []
        using_original_classification = True
        using_only_one_classification = False
        using_count_pre_omicron_classification = False
        using_count_omicron_classification = False
        folder= 'Sheraton_mar_23_2023'
        fname = 'not_in_Ours.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df_a  = pd.read_excel(fname)

        #The end of the study (EOS) date
        ub_date = datetime.datetime(2022,9,13)

        #The starting date for Omicron BA.5
        lb_date = datetime.datetime(2022,7,1)

        ub_doe = ub_date
        lb_dor = lb_date

        #When did the quantification of the risk start?
        study_start_date = lb_date
        study_end_date = ub_date

        vac_dates_h = self.parent.LIS_obj.vaccine_date_cols
        vac_types_h = self.parent.LIS_obj.vaccine_type_cols
        inf_dates_h = self.parent.LIS_obj.positive_date_cols
        inf_waves_h = self.parent.LIS_obj.wave_of_inf_cols
        counter = 0

        EVT = 'Outcome'
        self.parent.df[EVT] = np.nan

        TSTE = 'TimeFromStartToEnd'
        #This is the time-to-event
        #The risk is measured from the
        #beginning of the Omicron BA.5 period.
        self.parent.df[TSTE] = np.nan

        TVTS = 'TimeFromVac4ToStart'
        #This is a covariate. Note that the
        #risk period is disjoint from this period.
        self.parent.df[TVTS] = np.nan

        EOS = 'RemovedBeforeStudyFinished'
        self.parent.df[EOS] = False

        ILV = 'InfectionLevel'
        self.parent.df[ILV] = np.nan

        VLV = 'VaccineLevel'
        self.parent.df[VLV] = np.nan

        T4L = 'TimeFromV4Level'
        #This variable is used to convert the
        #TimeFromVac4ToStart variable into
        #a categorical variable.
        self.parent.df[T4L] = np.nan

        TFI = 'TimeFromInfection'
        self.parent.df[TFI] = np.nan

        OMC = 'OmicronCount'
        self.parent.df[OMC] = np.nan

        #Categories
        TFIL = 'TimeFromInfectionLevel'
        self.parent.df[TFIL] = np.nan
        big_M = 1000

        DOEOE = 'DateOfEndOrEvent'
        self.parent.df[DOEOE] = np.nan

        combined = 0

        N = self.parent.df.shape[0]
        iterator = pbar(self.parent.df.iterrows(), total=N)
        for index_m, row_m in iterator:

            ID = row_m['ID']

            site = row_m['Site']
            #Exclude anyone from sites 20 or 61.
            if site in [20,61]:
                continue

            #Exclude anyone that entered the study on/after Sept 13 2022
            doe = row_m['Enrollment Date']
            if pd.notnull(doe):
                if ub_doe <= doe:
                    is_in_df_a = df_a['ID'] == ID
                    #If this participant is in Ahmad's file
                    #then the enrollment day is not really a 
                    #problem since we have authorization
                    #to use the history of the patient
                    if is_in_df_a.any():
                        pass
                    else:
                        continue
            dor = row_m['Date Removed from Study']

            #Exclude anyone that was discharged/dead before July 01 2022
            if pd.notnull(dor):
                if dor < lb_dor:
                    continue

            vac_dates = row_m[vac_dates_h]
            vac_types = row_m[vac_types_h]

            #Exclude anyone that does not have 4 doses
            if vac_dates.count() < 4:
                continue

            vac_dates = vac_dates.iloc[:4]
            vac_types = vac_types.iloc[:4]

            #Exclude anyone that does not have 4 doses of mRNA vaccine
            if not vac_types.isin(['Moderna','Pfizer']).all():
                continue

            if vac_types.isin(['Pfizer']).all():
                #All Pfizer?
                #self.parent.df.loc[index_m, VLV] = 'A'
                self.parent.df.loc[index_m, VLV] = 'PfizerAll'
            elif vac_types.isin(['Moderna']).all():
                #All Moderna?
                #self.parent.df.loc[index_m, VLV] = 'B'
                self.parent.df.loc[index_m, VLV] = 'ModernaAll'
            else:
                #Combination
                #self.parent.df.loc[index_m, VLV] = 'C'
                self.parent.df.loc[index_m, VLV] = 'Mixed'

            vac_date_4 = vac_dates[-1]
            #Exclude anyone that does not have 4 doses by July 01 2022
            if study_start_date < vac_date_4:
                continue

            #Another covariate -> Time from vaccination 
            #to the start of the study
            delta_start_vac4 = study_start_date - vac_date_4
            delta_start_vac4 = delta_start_vac4.days
            if delta_start_vac4 < 0:
                raise ValueError('Negative delta start vac4')
            self.parent.df.loc[index_m, TVTS] = delta_start_vac4

            inf_dates = row_m[inf_dates_h]
            had_the_event = False

            if inf_dates.count() == 0:
                #Never infected ==> Infection level A
                #self.parent.df.loc[index_m, ILV] = 'A'
                self.parent.df.loc[index_m, ILV] = 'NoPriorInf'
                self.parent.df.loc[index_m, TFI] = big_M
            else:
                #s = inf_dates.notnull()
                #inf_dates = inf_dates[s]

                #Infection window =  Between July 1st and Sept 13 2022
                s1 = study_start_date <= inf_dates
                s2 = inf_dates <= ub_date
                s = s1 & s2
                inf_dates = inf_dates[s]
                if 0 < len(inf_dates):
                    #We have an event.
                    had_the_event = True
                    inf_date = inf_dates.iloc[0]

                    if pd.notnull(dor) and dor < inf_date:
                        raise ValueError('Removed before infection')

                    self.parent.df.loc[index_m, DOEOE] = inf_date

                    delta_inf_start = inf_date - study_start_date
                    delta_inf_vac4 = inf_date - vac_date_4

                    delta_inf_start = delta_inf_start.days
                    delta_inf_vac4 = delta_inf_vac4.days

                    self.parent.df.loc[index_m, EVT] = had_the_event
                    self.parent.df.loc[index_m, TSTE] = delta_inf_start

                    if delta_inf_start < 0 or delta_inf_vac4 < 0:
                        raise ValueError('Unexpected vac date #4')
                    #Exclude anyone that developed
                    #the outcome within 7 days of 4th vaccine dose
                    if delta_inf_vac4 <= 7:
                        continue

                #We are going to define the ILEV (infection level)
                omicron_start_date = datetime.datetime(2021,12,15)
                omicron_end_date   = datetime.datetime(2022,7,1)
                inf_dates = row_m[inf_dates_h]
                #Chronological
                inf_dates = inf_dates.sort_values()

                #Do we have infections before the Omicron BA.5 period?
                s = inf_dates < study_start_date
                n_inf_before_O5 = s.sum()

                if n_inf_before_O5 == 0:
                    #Never infected ==> Infection level A
                    #self.parent.df.loc[index_m, ILV] = 'A'
                    self.parent.df.loc[index_m, ILV] = 'NoPriorInf'
                    self.parent.df.loc[index_m, TFI] = big_M
                else:
                    #----------Time from last infection to
                    #----------the start of the study
                    #print(inf_dates.loc[s])
                    #print(inf_dates.loc[s].iloc[-1])
                    delta_inf = study_start_date - inf_dates.loc[s].iloc[-1]
                    delta_inf = delta_inf / np.timedelta64(1,'D')
                    #print(f'{delta_inf=}')
                    self.parent.df.loc[index_m, TFI] = delta_inf
                    #print('---------------------')



                    pre_omicron_inf   = inf_dates < omicron_start_date
                    n_pre_omicron_inf = pre_omicron_inf.sum()

                    s1 = omicron_start_date <= inf_dates
                    s2 = inf_dates < omicron_end_date
                    omicron_inf = s1 & s2
                    n_omicron_inf = omicron_inf.sum()

                    if n_omicron_inf == 2:
                        self.parent.df.loc[index_m, OMC] = 'Double Omicron'
                    elif n_omicron_inf == 1:
                        self.parent.df.loc[index_m, OMC] = 'Single Omicron'

                    if using_original_classification:
                        if 1 < n_pre_omicron_inf + n_omicron_inf:
                            #Multiple infections ==> Infection level D
                            #self.parent.df.loc[index_m, ILV] = 'D'
                            self.parent.df.loc[index_m, ILV] = 'Multiple'
                        elif n_pre_omicron_inf == 1:
                            #Exactly One pre-omicron ==> Infection level B
                            self.parent.df.loc[index_m, ILV] = 'OnePreOmicron'
                        else:
                            #Exactly One omicron ==> Infection level C
                            self.parent.df.loc[index_m, ILV] = 'OneOmicron'
                            if n_omicron_inf != 1:
                                raise ValueError('Unexpected value for Omicron')

                    elif using_only_one_classification:
                        if 1 <= n_omicron_inf and 1 <= n_pre_omicron_inf:
                            self.parent.df.loc[index_m, ILV] = 'Mixed'
                        elif n_pre_omicron_inf == 0:
                            self.parent.df.loc[index_m, ILV] = 'OnlyOmicron'
                        else:
                            self.parent.df.loc[index_m, ILV] = 'OnlyPreOmicron'
                            if n_omicron_inf != 0:
                                raise ValueError('Unexpected value for Omicron')

                    elif using_count_pre_omicron_classification:
                        if 0 < n_omicron_inf:
                            self.parent.df.loc[index_m, ILV] = 'HasOmicron'
                        elif n_pre_omicron_inf == 1:
                            self.parent.df.loc[index_m, ILV] = 'Only1PreOmicron'
                        else:
                            self.parent.df.loc[index_m, ILV] = 'Only2PreOmicron'
                            if n_pre_omicron_inf != 2:
                                raise ValueError('Unexpected value for PreOmicron')

                    elif using_count_omicron_classification:
                        #print(f'{n_pre_omicron_inf=}, {n_omicron_inf=}')
                        if n_pre_omicron_inf == 2 and n_omicron_inf == 1:
                            print('2P+1O')
                        if n_pre_omicron_inf == 1 and n_omicron_inf == 2:
                            print('1P+2O')
                        if n_pre_omicron_inf + n_omicron_inf > 2:
                            print('>2')
                        if 0 < n_pre_omicron_inf:
                            self.parent.df.loc[index_m, ILV] = 'HasPreOmicron'
                        elif n_omicron_inf == 1:
                            self.parent.df.loc[index_m, ILV] = 'Only1Omicron'
                        else:
                            self.parent.df.loc[index_m, ILV] = 'Only2Omicron'
                            #print(f'{had_the_event=}')
                            if n_omicron_inf != 2:
                                raise ValueError('Unexpected value for Omicron')

                    #elif using_count_one_and_one_classification:
                        #if n_pre_omicron_inf == 1 and n_omicron_inf == 1:
                            #self.parent.df.loc[index_m, ILV] = 'Has1P1O'
                        #elif n_pre_omicron_inf == 2 and n_omicron_inf == 1:
                            #self.parent.df.loc[index_m, ILV] = 'Has2P1O'
                        #else:
                            #self.parent.df.loc[index_m, ILV] = 'Only2Omicron'
                            ##print(f'{had_the_event=}')
                            #if n_omicron_inf != 2:
                                #raise ValueError('Unexpected value for Omicron')

            if not had_the_event:
                #No infection within the given window.
                #had_the_event = False
                delta_end_start = study_end_date - study_start_date
                delta_end_start = delta_end_start.days
                self.parent.df.loc[index_m, DOEOE] = study_end_date
                if delta_end_start < 0:
                    raise ValueError('Unexpected delta end minus start')
                if pd.notnull(dor):
                    if dor < study_end_date:
                        #Removed before the end of the study
                        delta_end_start = dor - study_start_date
                        delta_end_start = delta_end_start.days
                        self.parent.df.loc[index_m, EOS] = True
                        self.parent.df.loc[index_m, DOEOE] = dor

                self.parent.df.loc[index_m, EVT] = False
                self.parent.df.loc[index_m, TSTE] = delta_end_start


            counter += 1
            L.append(index_m)

        print(f'{counter=}')

        STP = 'SiteType'
        self.parent.df[STP] = 'LTC'
        self.parent.df[STP] = self.parent.df[STP].where(self.parent.df['Site'] < 50, 'RH')

        cols = ['ID', STP, 'Age', 'Sex', EVT, TSTE, EOS, ILV, VLV]

        df = self.parent.df.loc[L,:].copy()
        #df = self.parent.df.loc[L,cols].copy()

        #print(len(df))
        #df.dropna(inplace=True)
        #print(len(df))

        #Add one day to avoid having a zero
        #for the time-to-event.
        #Recommended by Ahmad.
        df[TSTE] += 1

        #Add one day to avoid having a zero
        #for the time-to-event.
        #Recommended by Ahmad.
        #TVTS = 'TimeFromVac4ToStart'
        df[TVTS] += 1

        #TFI = 'TimeFromInfection'
        df[TFI] += 1

        intervals = [0, 30, 60, 90, 180, 1000, 1002]
        intervals = [0, 90, 180, 1000, 1002]
        intervals = [0, 30, 90, 180, 1002]

        labels = []
        for k in range(len(intervals)-1):
            a = intervals[k]
            b = intervals[k+1]
            txt = 'From' + str(a) + 'To' + str(b)
            #if a == 1000:
                #txt = 'NoInfection'
            labels.append(txt)

        bins = pd.cut(df[TFI], intervals, labels=labels)
        print(bins)


        df[TFIL] = bins
        vc = df[TFIL].value_counts().sort_index()
        fig,ax = plt.subplots()
        vc.plot(kind='barh', ax = ax)
        txt = 'Days from last inf. to the start of the study'
        ax.set_title(txt)
        ax.set_xlabel('# of participants')

        fname = 'days_from_last_inf_to_study_start.png'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        fig.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close('all')



        s = df[TVTS].describe()
        #print(s)
        #The rows are:
        #0 Count
        #1 Mean
        #2 STD
        #3 Min
        #4 25%
        #5 50%
        #6 75%
        #7 Max
        #We use the quartiles of the variable
        #Time from Vaccination to the Start of
        #the Study to create a categorical variable.
        intervals = s[4:].astype(int).to_list()
        intervals = [0] + intervals

        #intervals = [0, 130, 150, 160, 260]
        #intervals = [0, 65, 130, 195, 260]

        labels = []
        for k in range(len(intervals)-1):
            a = intervals[k]
            b = intervals[k+1]
            txt = 'From' + str(a) + 'To' + str(b)
            labels.append(txt)

        bins = pd.cut(df[TVTS], intervals, labels=labels)

        #bins = pd.cut(df[TSTE], intervals)

        df[T4L] = bins

        df[EVT] = df[EVT].apply(lambda x: 1 if x else 0)

        s1 = slice('Wave of Inf 1','Blood Draw:Repeat - JR')
        s2 = slice('Had a PCR+','Notes/Comments')
        c1 = df.loc[:,s1].columns.to_list()
        c2 = df.loc[:,s2].columns.to_list()
        c = c1 + c2
        df.drop(columns=c, inplace=True)




        #for m in multi_omicron:
            #print(m)
        #print(df)
        folder= 'Sheraton_mar_23_2023'
        fname = 'list_of_participants_for_covid_ltc_003.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df.to_excel(fname, index=False)

    def add_outbreak_to_list(self):
        #Mar 23 2023
        folder= 'Sheraton_mar_23_2023'
        fname = 'ahmad.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df_a = pd.read_excel(fname)
        df_a = df_a[['ID','outbreak_count']].copy()
        #df_a.rename(columns={'outbreak_count':'Outbreaks'}, inplace=True)
        df_a['Outbreaks'] = 'SixOrLess'
        s = df_a['outbreak_count'] == '<=6'
        df_a['Outbreaks'] = df_a['Outbreaks'].where(s, 'MoreThanSix')
        df_a.drop(columns='outbreak_count', inplace=True)

        fname = 'list_of_participants_for_covid_ltc_003.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df_m = pd.read_excel(fname)

        df = pd.merge(df_m, df_a, how='inner', on='ID')
        fname = 'LTC003_list.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df.to_excel(fname)


    def create_design_matrix_sheraton(self):
        #Mar 23 2023
        folder= 'Sheraton_mar_23_2023'
        #fname = 'list_of_participants_for_covid_ltc_003.xlsx'
        fname = 'LTC003_list.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df = pd.read_excel(fname)
        EVT = 'Outcome'
        AGE = 'Age'
        TSTE = 'TimeFromStartToEnd'
        EOS = 'RemovedBeforeStudyFinished'
        ILV = 'InfectionLevel'
        VLV = 'VaccineLevel'
        T4L = 'TimeFromV4Level'
        SEX = 'Sex'
        STP = 'SiteType'
        OUT = 'Outbreaks'
        TFI = 'TimeFromInfection'
        TFIL = 'TimeFromInfectionLevel'
        TFIL_N = TFIL + '_NoInfection'
        TFIL_N = TFIL + '_From180To1002'

        #df[EVT] = df[EVT].apply(lambda x: 1 if x else 0)

        cols = [STP, SEX, ILV, VLV, T4L, TFIL, OUT]
        X = pd.get_dummies(df[cols])
        cols_to_remove = [
                'SiteType_LTC',
                'Sex_Male',
                'InfectionLevel_NoPriorInf',
                'VaccineLevel_PfizerAll',
                'TimeFromV4Level_From0To131',
                TFIL_N,
                'Outbreaks_SixOrLess']
        X.drop(columns = cols_to_remove, inplace=True)
        Z = df[[EVT,AGE,TSTE]]
        M = pd.concat([Z,X], axis=1)

        fname = 'design_matrix.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        M.to_excel(fname, index=False)
        #cols = ['ID','Age', EVT, TSTE, EOS, ILV, VLV]

    def survival_analysis_sheraton(self):
        folder= 'Sheraton_mar_23_2023'
        fname = 'design_matrix.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df = pd.read_excel(fname)


        fname = 'survival_analysis_labels.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df_labels = pd.read_excel(fname)
        dc_old_to_new = {}

        for _, row in df_labels.iterrows():
            old = row['Old']
            new = row['New']
            dc_old_to_new[old]=new

        plot_hazard_ratios= True
        remove_time_level = False


        EVT = 'Outcome'
        AGE = 'Age'
        TSTE = 'TimeFromStartToEnd'
        EOS = 'RemovedBeforeStudyFinished'
        ILV = 'InfectionLevel'
        VLV = 'VaccineLevel'
        T4L = 'TimeFromV4Level'
        SEX = 'Sex'
        STP = 'SiteType'
        OUT = 'Outbreaks'
        TFI = 'TimeFromInfection'
        TFIL = 'TimeFromInfectionLevel'
        TFIL_N = TFIL + '_NoInfection'

        #s1 = df[TSTE] <= 8
        #s2 = 0 <= df[TSTE]
        #s  = s1 & s2
        #df = df.loc[s]

        #s1 = df[TSTE] <= 29
        #s2 = 9 <= df[TSTE]
        #s  = s1 & s2
        #df = df.loc[s]

        #s1 = df[TSTE] <= 76
        #s2 = 30 <= df[TSTE]
        #s  = s1 & s2
        #df = df.loc[s]


        if remove_time_level:
            R = []
            for col in df.columns:
                if col.startswith(T4L):
                    R.append(col)
            if 0 < len(R):
                df.drop(columns=R, inplace=True)

        self.cph = CoxPHFitter()

        if remove_time_level is False:
            self.cph.fit(df, TSTE, EVT, strata=['TimeFromV4Level_From155To163'])
        else:
            self.cph.fit(df, TSTE, EVT)
        S = self.cph.summary
        S.rename(index=dc_old_to_new, inplace=True)
        fname = 'survival_analysis_summary.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        S.to_excel(fname)
        self.cph.print_summary(model='LTC003')
        #self.cph.print_summary(model='LTC003', style='latex')
        self.cph.check_assumptions(df)

        if plot_hazard_ratios:
            fig,ax = plt.subplots()
            X = S['exp(coef)']
            lower_delta = X - S['exp(coef) lower 95%']
            upper_delta = S['exp(coef) upper 95%'] - X
            errors = pd.concat([lower_delta, upper_delta], axis=1)
            errors = errors.values.T
            #print(errors)
            p_values = S['p']
            ax.barh(X.index,X,color='b', xerr=errors,
                    error_kw=dict(ecolor='red', lw=2,))
            ax.set_xlabel('Hazard ratio')
            for k,c in enumerate(X):
                y_pos = k
                x_pos = c
                p_value = p_values.iloc[k]
                #print(f'{p_value=}')
                if c < 0:
                    x_pos = 0
                elif 5 < c:
                    x_pos = 0.9*c
                    y_pos *= 1.15
                else:
                    x_pos = S['exp(coef) upper 95%'][k]
                txt = '{:.2f}'.format(c)
                if p_value < 0.001:
                    txt += '***'
                elif p_value < 0.01:
                    txt += '**'
                elif p_value < 0.05:
                    txt += '*'
                ax.text(x_pos, y_pos, txt, ha='left', fontsize=16)

            fname = 'coeff_march_13_2023.png'
            fname = os.path.join(self.parent.requests_path, folder, fname)
            #ax.tick_params(axis='x', rotation=90)
            fig.savefig(fname, bbox_inches='tight', pad_inches=0)

    def plot_population_statistics_sheraton(self):

        folder= 'Sheraton_mar_23_2023'
        main_folder = folder
        fname = 'LTC003_list.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df = pd.read_excel(fname)

        def map_interval_to_Q(x):
            if x == 'From0To131':
                return 'Q1'
            elif x == 'From131To155':
                return 'Q2'
            elif x == 'From155To163':
                return 'Q3'
            elif x == 'From163To186':
                return 'Q4'
            else:
                raise ValueError('Unexpected')

        df['TimeFromV4Level'] = df['TimeFromV4Level'].apply(map_interval_to_Q)
        TS4D = 'DaysFrom4thDoseToStart'
        df.rename(columns={'TimeFromV4Level':TS4D}, inplace=True)

        #u = df['Site'].unique()
        #w = sum(u < 50)
        #print(u)
        #print(w)

        plot_stats = True

        L = ['Age', 'Sex', 'InfectionLevel', 'VaccineLevel',
                'SiteType','Outbreaks', TS4D]
        use_bar = {'Age':False, 'Sex':True,
                'InfectionLevel':True, 'VaccineLevel':True,
                'SiteType':True, 'Outbreaks':True, TS4D:True}
        event = 'Outcome'

        for var in L:
            if plot_stats:
                fig, ax = plt.subplots()
            if use_bar[var]:
                prop = 'Proportion'
                vc = df[var].groupby(df[event]).value_counts(normalize=False)
                labels = vc.loc[0].index.to_list()
                print(labels)
                print(vc)
                T = vc.unstack(level=0)
                print(T.values)
                chi_sq_stat, p_value, dofs, expected = chi_sq_test(T.values)
                print(f'{chi_sq_stat=}')
                print(f'{p_value=}')
                print(f'{dofs=}')
                S = df[var].groupby(df[event]).value_counts(normalize=True)
                S = S.rename(prop)
                S = S.reset_index()

                if plot_stats:
                    sns.barplot(ax = ax,
                            x=event, y=prop,
                            data=S,
                            hue = var)
                    for cat_index, container in enumerate(ax.containers):
                        for event_index, R in enumerate(container):
                            label = labels[cat_index]

                            #Recall that people with 2 omicron only appear in the 
                            #Outcome = 0 group.
                            if label == 'Only2Omicron' and event_index == 1:
                                continue

                            #print(cat_index)
                            #print(event_index)
                            #print(R)
                            #print(labels[cat_index])
                            h = R.get_height()
                            w = R.get_width()/2
                            xy = R.get_xy()
                            h = round(h*100)/100
                            R.set_height(h)

                            #print(f'{event_index=}')
                            #print(vc.loc[event_index].index[cat_index])
                            count = vc.loc[(event_index, label)]
                            #print(f'{count=}')

                            #Center the number label
                            hp = h/2

                            #To avoid clustering on the bottom.
                            hp -= 0.025

                            if count < 5:
                                hp += 0.05
                                hp += 0.05
                            ax.text(xy[0]+w, hp, count, ha='center', fontsize=16)


                        #Note that the container was modified by setting
                        #R.set_height(h)
                        ax.bar_label(container)
            else:
                if plot_stats:
                    s = df[event] == 1
                    a = df.loc[s,var]
                    b = df.loc[~s,var]
                    U,p_value = MannWhitney(a,b)
                    print('Mann-Whitney')
                    print(f'{p_value=}')
                    sns.violinplot(x=event, y=var, data=df, ax = ax, cut=0)
                    ax.text(0.5, 90, 615, ha='center', fontsize=16)
                    ax.text(0.7, 90, 133, ha='center', fontsize=16)

            if p_value < 0.001:
                p_label = 'p<0.001'
            elif p_value < 0.01:
                p_label = 'p<0.01'
            elif p_value < 0.05:
                p_label = 'p<0.05'
            else:
                p_label = 'p={:.2f}'.format(p_value)
            print('============================')

            stats_folder = 'stats'
            fname = var.replace('?','')
            fname += '.png'
            fname = os.path.join(self.parent.requests_path,
                    main_folder,
                    stats_folder,
                    fname)
            if plot_stats:
                p_label = '$' + p_label + '$'
                plt.title(p_label, fontsize=18)
                fig.savefig(fname, bbox_inches='tight', pad_inches=0)
                plt.close('all')


    def plot_kaplan_meier_sheraton(self):
        km_folder = 'km'
        EVT = 'Outcome'
        AGE = 'Age'
        TSTE = 'TimeFromStartToEnd'
        EOS = 'RemovedBeforeStudyFinished'
        ILV = 'InfectionLevel'
        VLV = 'VaccineLevel'
        T4L = 'TimeFromV4Level'
        SEX = 'Sex'
        STP = 'SiteType'
        OUT = 'Outbreaks'

        folder= 'Sheraton_mar_23_2023'
        fname = 'LTC003_list.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df = pd.read_excel(fname)


        df_g = df.groupby('InfectionLevel')
        km  = KMFitter()

        groups = ['NoPriorInf','Multiple','OnePreOmicron','OneOmicron']
        colors = ['blue','green','gray','orange']
        group_to_color = {}

        for g,c in zip(groups,colors):
            group_to_color[g] = c

        ref_group = 'NoPriorInf'
        ref_group = 'OneOmicron'
        df_ref = df_g.get_group(ref_group)
        s_ref_group = df['InfectionLevel'] == ref_group

        for group, df in df_g:

            if group == ref_group:
                continue


            print('Log_rank_test between ', group, ' and ', ref_group)
            log_rank_test = LRT(df[TSTE],df_ref[TSTE], df[EVT], df_ref[EVT])
            log_rank_test.print_summary()
            print(log_rank_test.p_value)

            fig,ax = plt.subplots()

            km.fit(df_ref[TSTE], df_ref[EVT], label=ref_group)
            km.plot_survival_function(ax=ax, color=group_to_color[ref_group])

            km.fit(df[TSTE], df[EVT], label=group)
            km.plot_survival_function(ax=ax, color=group_to_color[group])

            ax.set_ylim([0.3,1])
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Survival(t)')

            fname = 'km_' + group + '.png'
            fname = os.path.join(self.parent.requests_path,
                                 folder, km_folder, fname)
            #ax.tick_params(axis='x', rotation=90)
            fig.savefig(fname, bbox_inches='tight', pad_inches=0)
            plt.close('all')
            print('========================')

        fig,ax = plt.subplots()

        for k, group in enumerate(groups):

            df = df_g.get_group(group)
            km.fit(df[TSTE], df[EVT], label=group)
            km.plot_survival_function(ax=ax, color=colors[k])


        fname = 'km_all.png'
        fname = os.path.join(self.parent.requests_path, folder, km_folder, fname)
        ax.set_ylim([0.3,1])
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Survival(t)')
        fig.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close('all')

    def compare_tables_for_sheraton(self):

        write_over_local_list = False
        write_over_foreign_list = True

        folder= 'Sheraton_mar_23_2023'
        fname = 'list_of_participants_for_covid_ltc_003.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df_m = pd.read_excel(fname)

        fname = 'ahmad.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df_a = pd.read_excel(fname)

        if write_over_local_list:
            w = df_m['ID'].isin(df_a['ID'])
            df_m['InAhmads?'] = w

            fname = 'LTC003_list.xlsx'
            fname = os.path.join(self.parent.requests_path, folder, fname)
            df_m.to_excel(fname)

        if write_over_foreign_list:
            w = df_a['ID'].isin(df_m['ID'])
            df_a['InTest?'] = w

            missing = df_a.loc[~w,'ID']

            fname = 'A_list.xlsx'
            fname = os.path.join(self.parent.requests_path, folder, fname)
            df_a.to_excel(fname)

            s = self.parent.df['ID'].isin(missing)
            df_t = self.parent.df[s]

            fname = 'not_in_our_list.xlsx'
            fname = os.path.join(self.parent.requests_path, folder, fname)
            df_t.to_excel(fname)

    def investigate_double_o_sheraton(self):
        fname = 'double_o.xlsx'
        folder= 'Sheraton_mar_23_2023'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df_so = pd.read_excel(fname)
        def get_site(txt):
            return txt[:2]
        df_so['Site'] = df_so['ID'].apply(get_site)
        vc = df_so['Site'].value_counts()
        fig, ax = plt.subplots()
        vc.plot(kind='barh', ax = ax)
        ax.set_xlabel('# of participants')
        ax.set_ylabel('Site')
        txt = 'Double Omicron (BA.1,BA.2) infection'
        n = vc.sum()
        txt += ' (n=' + str(n) + ')'
        ax.set_title(txt)

        fname = 'double_o_dist.png'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        fig.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close('all')

    def draw_cloud_history_from_serology_for_sheraton(self):
        #This function is used to plot the serology trajectory
        #of a given participant.

        #The original version of this function was created to
        #satisfy the request stipulated on the folder:
        #main_folder = 'Tara_feb_21_2023'

        main_folder = 'Sheraton_mar_23_2023'
        bio_list = ['Nuc-IgG-100']
        bio_list += ['RBD-IgG-100']
        bio_list += ['Nuc-IgA-100']
        bio_list += ['Spike-IgG-100']
        bio_list += ['Wuhan (SB3)']
        bio_list += ['Omicron (BA.1)']
        date_format = mpl.dates.DateFormatter('%b-%y')
        study_start_date = datetime.datetime(2022,7,1)
        study_end_date = datetime.datetime(2022,9,13)

        #The serology thresholds are stored in the
        #following folder.
        fname = 'serology_thresholds.xlsx'
        fname = os.path.join(self.parent.LSM_path, fname)
        df_t = pd.read_excel(fname)

        #List of those participants that had a double
        #omicron infection (before BA.5).
        fname = 'double_o.xlsx'
        folder= 'Sheraton_mar_23_2023'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df_do = pd.read_excel(fname)

        #List of those participants that had a single
        #omicron infection (before BA.5).
        fname = 'single_o.xlsx'
        folder= 'Sheraton_mar_23_2023'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df_so = pd.read_excel(fname)

        #Filter the Master data frame for the double o individuals.
        s_double = self.parent.df['ID'].isin(df_do['ID'])
        s_single = self.parent.df['ID'].isin(df_so['ID'])

        df_m = self.parent.df.copy()
        df_m['D'] = s_double
        df_m['S'] = s_single

        #The serology thresholds are stored in the
        #following folder.
        fname = 'serology_thresholds.xlsx'
        fname = os.path.join(self.parent.LSM_path, fname)
        df_t = pd.read_excel(fname)

        inf_date_h = self.parent.LIS_obj.positive_date_cols
        vac_date_h = self.parent.LIS_obj.vaccine_date_cols

        #The counter is simply used to 
        #manually limit the number
        #of iterations.
        counter = 0

        N = df_m.shape[0]


        for bio in bio_list:

            #One single plot
            fig, ax = plt.subplots()

            for index_m, row_m in pbar(df_m.iterrows(), total=N):
                counter += 1
                if row_m['D']:
                    cloud_color='blue'
                    inf_color='yellow'
                    inf_style='solid'
                elif row_m['S']:
                    cloud_color='red'
                    inf_color='cyan'
                    inf_style='solid'
                else:
                    continue

                #if 2 < counter:
                    #break
                ID = row_m['ID']
                s = self.parent.LSM_obj.df['ID'] == ID
                if not s.any():
                    continue
                df_s = self.parent.LSM_obj.df[s]

                #This list will store the date of collection
                #and the value of the biological parameter.
                L = []

                s = df_t['Ig'] == bio
                threshold = df_t.loc[s,'Threshold'].iloc[0]
                #Create a list of the value of
                #the biological parameter and the 
                #date of collection.
                for index_s, row_s in df_s.iterrows():
                    bio_value = row_s[bio]
                    if pd.notnull(bio_value):
                        doc = row_s['Date Collected']
                        L.append((doc, bio_value))
                #At least two points
                if 1 < len(L):

                    #Create DF.
                    df = pd.DataFrame(L, columns=['Time',bio])

                    #Was the threshold crossed?
                    flag_crossed_threhold = False
                    if df[bio].min() < threshold and threshold < df[bio].max():
                        flag_crossed_threhold = True
                    else:
                        #Did not cross.
                        flag_below = False
                        if df[bio].min() < threshold:
                            flag_below = True

                    #Serology time
                    df.plot(ax=ax, x='Time',
                            linestyle='None',
                            y=bio, marker='o',
                            alpha = 0.6,
                            color=cloud_color)

                    #Infection time
                    flag_had_infections = False
                    inf_dates = row_m[inf_date_h]

                    if 0 < inf_dates.count():
                        flag_had_infections = True
                        s = inf_dates.notnull()
                        inf_dates = inf_dates[s]

                        flag_infection_within = False

                        #Iterate over infection times.
                        for _ , inf_date in inf_dates.items():
                            ax.axvline(inf_date, color=inf_color,
                                       linestyle=inf_style, linewidth=1)
                            #If there was an infection between the 
                            #earliest time of collection and the
                            #latest time of collection, then we had
                            #an infection within.
                            c1 = df['Time'].min() <= inf_date
                            c2 = inf_date <= df['Time'].max()
                            if c1 and c2:
                                flag_infection_within = True

                    #Vaccination time
                    vac_dates = row_m[vac_date_h]
                    if 0 < vac_dates.count():
                        s = vac_dates.notnull()
                        vac_dates = vac_dates[s]
                        for k,(_ , date) in enumerate(vac_dates.items()):
                            if k == -3:
                                ax.axvline(date, color='black', linestyle='-', linewidth=1)



            #Plot threshold line
            ax.axhline(threshold, color='gray', linewidth=3)

            #Plot study start and end date lines
            ax.axvline(study_start_date, color='brown',
                       linestyle='dashdot', linewidth=3)
            ax.axvline(study_end_date, color='brown',
                       linestyle='dashdot', linewidth=3)

            ax.xaxis.set_major_formatter(date_format)
            ax.set_ylabel(bio)
            ax.get_legend().remove()
            bio = bio.replace('.','_')
            bio = bio.replace(' ','_')
            fname = 'cloud_' + bio + '.png'
            stats_folder = 'stats'
            storage_folder = os.path.join(self.requests_path,
                                          main_folder,
                                          stats_folder)
            if not os.path.exists(storage_folder):
                os.makedirs(storage_folder)
            fname = os.path.join(storage_folder, fname)
            fig.savefig(fname, bbox_inches='tight', pad_inches=0)
            plt.close('all')


    def compute_concordance_for_sheraton(self):

        EVT = 'Outcome'
        AGE = 'Age'
        TSTE = 'TimeFromStartToEnd'
        EOS = 'RemovedBeforeStudyFinished'
        ILV = 'InfectionLevel'
        VLV = 'VaccineLevel'
        T4L = 'TimeFromV4Level'
        SEX = 'Sex'
        STP = 'SiteType'
        OUT = 'Outbreaks'
        HZR = 'Hazard'

        folder= 'Sheraton_mar_23_2023'
        fname = 'design_matrix.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df = pd.read_excel(fname)
        print("==================")
        print(self.cph.params_)
        #print(self.cph.baseline_hazard_)
        hazard = self.cph.predict_partial_hazard(df)
        df[HZR] = hazard
        df = df.sort_values(TSTE)
        df = df.reset_index()
        n_rows = df.shape[0]
        comparable = set()
        concordant = set()
        for k in range(n_rows):
            t1 = df.loc[k,TSTE]
            e1 = df.loc[k,EVT]
            h1 = df.loc[k,HZR]
            for i in range(k+1,n_rows):
                t2 = df.loc[i,TSTE]
                h2 = df.loc[i,HZR]
                if t1 < t2 and e1 == 1:
                    comparable.add((k,i))
                    if h2 < h1:
                        concordant.add((k,i))
        n_comp = len(comparable)
        n_conc = len(concordant)
        print('# of comparable:', n_comp)
        print('# of concordant:', n_conc)
        R = n_conc / n_comp
        print('concordance idx:', R)

    def dawns_request_mar_04_2023(self):
        #Dawn requested a graphic that illustrates the following.
        #Out of all individuals that died from X in 2020, how many
        #of those were aged 70 or more?
        #X = {Cancer, Heart disease, chronic lower respiratory diseases}


        COD = 'Cause of death (ICD-10)'
        DAT = 'REF_DATE'
        AGE = 'Age at time of death'
        CHR = 'Characteristics'
        VAL = 'VALUE'
        NOD = 'Number of deaths'

        folder = 'Dawn_mar_04_2023'
        fname = 'stats_canada.xlsx'
        fname = os.path.join(self.requests_path, folder, fname)
        df_c = pd.read_excel(fname, sheet_name='COD')
        print(df_c)

        fname = 'stats_canada.xlsx'
        fname = os.path.join(self.requests_path, folder, fname)
        df = pd.read_excel(fname, sheet_name='data')
        rexp = re.compile(' \[.*')
        def get_code(txt):
            obj = rexp.search(txt)
            if obj:
                x = obj.group(0)
                y = txt.replace(x,'')
                #print(y)
                return y
            else:
                return txt

        df[COD] = df[COD].apply(get_code)
        age_groups = df[AGE].unique()
        all_ages = age_groups[0]
        age_groups_s = age_groups[1:]
        df_g = df.groupby([DAT, AGE, CHR])

        #q = df_g.loc[(2020,'Age at time of death, 65 to 69 years','Number of deaths'),'VALUE']
        #g = df_g.get_group((2020,'Age at time of death, 65 to 69 years','Number of deaths'))
        #print(g[[COD,VAL]])

        #Cause of death to count
        cod_to_count = {}
        cod_to_total = {}
        #Iterate over death causes
        for index, row in df_c.iterrows():
           cod = row[COD]
           cod_to_count[cod] = 0
           for age_group in age_groups_s:
               #print(f'{age_group=}')
               g = df_g.get_group((2020,age_group,NOD))
               s = g[COD] == cod
               v = g.loc[s, VAL].iloc[0]
               cod_to_count[cod] += v
           g = df_g.get_group((2020,all_ages,NOD))
           s = g[COD] == cod
           v = g.loc[s, VAL].iloc[0]
           cod_to_total[cod] = v

        df1 = pd.DataFrame.from_dict(cod_to_count, orient='index')
        df2 = pd.DataFrame.from_dict(cod_to_total, orient='index')
        m = pd.merge(df1,df2, left_index=True, right_index=True, how='outer')
        m.rename(columns={'0_x':'Counts', '0_y':'Total'}, inplace=True)
        m['Percentage'] = m['Counts'] / m['Total'] * 100
        m = m.sort_values('Percentage', ascending=False)
        print(m)
        fname = 'summary_stats_canada.xlsx'
        fname = os.path.join(self.requests_path, folder, fname)
        #m.to_excel(fname)

    def dawns_request_mar_05_2023(self):
        #Dawn requested a graphic that illustrates the following.
        #Out of all individuals that died from X in 2020, how many
        #of those were aged 65 or more?
        #X = {Cancer, Heart disease, chronic lower respiratory diseases}
        folder = 'Dawn_mar_04_2023'
        fname = 'summary_stats_canada.xlsx'
        fname = os.path.join(self.requests_path, folder, fname)
        df = pd.read_excel(fname, sheet_name = 'merged')
        df['Percentage'] = np.round(df['Percentage'])
        #df = df.sort_values('Percentage', ascending=False)

        fig, ax = plt.subplots()
        colors = df['Color']
        sns.barplot(ax = ax,
                x='Percentage',
                y='Cause of death',
                data = df,
                palette=colors)
        ax.bar_label(ax.containers[0])
        fname = 'plot.svg'
        fname = os.path.join(self.requests_path, folder, fname)
        print(fname)
        ax.set_ylabel('')
        fig.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close('all')

    def generate_report_for_resident_questionnaire(self):
        folder = 'Lindsay_apr_05_2023'
        folder2 = 'test'

        regexp_bracket = re.compile('\[(?P<value>.*)\]')
        regexp_dot_number = re.compile('[.]\d+')
        add_another_condition = 'Would you like to add another condition?'
        add_another_medication = 'Would you like to add another medication?'

        fname = 'column_types_for_resident_questionnaire.xlsx'
        fname = os.path.join(self.requests_path, folder, folder2, fname)
        df_t = pd.read_excel(fname, dtype=str)
        if df_t['Name'].value_counts().gt(1).any():
            vc = df_t['Name'].value_counts()
            s  = vc.gt(1)
            vc = vc[s]
            print(vc)
            raise ValueError('Non unique entries at Name in column type LTC')
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

        #Load the Resident Questionnaire database
        fname = 'resident_questionnaire.xlsx'
        fname = os.path.join(self.requests_path, folder, folder2, fname)
        df = pd.read_excel(fname, dtype = name_to_type, parse_dates=date_cols)
        #print(df)
        df_t.dropna(subset='Section', inplace=True)
        df_t = df_t.sort_values('Section', ascending=True, kind='stable')
        df_t = df_t.groupby('Section')

        SIT = 'Participant Site (2 digits, example: 08)'
        PID = 'Participant ID (7 digits, example: 0123456)'


        for index_m, row_m in df.iterrows():

            site = row_m[SIT]
            pid  = row_m[PID]

            if pd.isnull(site) or pd.isnull(pid):
                continue

            if len(site) != 2:
                raise ValueError('Site length')

            if len(pid) != 7:
                raise ValueError('PID length')

            ID = site + '-' + pid

            fname = ID + '.csv'
            fname = os.path.join(self.requests_path, folder, folder2, fname)

            with open(fname, 'w', newline='') as f:

                writer = csv.writer(f)

                for section, df_g in df_t:
                    writer.writerow([])
                    txt = 'Question #' + section
                    desc= df_g['Description'].iloc[0]
                    writer.writerow([txt, desc])

                    for index_s, row_s in df_g.iterrows():
                        col_name = row_s['Name']
                        value    = row_m[col_name]
                        dtype    = row_s['Type']

                        if pd.isnull(value):
                            continue

                        if dtype == 'date':
                            value = value.strftime('%d-%b-%Y')


                        #These questions allow for multiple responses.
                        #Ethnicity
                        #Adverse effects after dose X
                        #Chronic Conditions
                        if section in ['03','19.1','19.2','19.3','19.4','35']:
                            if value == 'No':
                                continue
                            else:
                                obj = regexp_bracket.search(col_name)
                                if obj:
                                    col_name = obj.group('value')
                                    writer.writerow([col_name, value])
                                    writer.writerow([])
                                    continue
                                else:
                                    raise ValueError('Error for bracket')

                        #Height and weight have descriptors
                        #within brackets.
                        #There is no Yes/No multiple response.
                        if section in ['20.a','20.b']:
                            obj = regexp_bracket.search(col_name)
                            if obj:
                                col_name = obj.group('value')
                                writer.writerow([col_name, value])
                                writer.writerow([])
                                continue
                            else:
                                raise ValueError('Error for bracket')


                        #Please enter the name of the condition
                        #Would you like to add another condition?
                        #Would you like to add another medication?
                        if add_another_condition in col_name:
                            continue
                        if add_another_medication in col_name:
                            continue
                        obj = regexp_dot_number.search(col_name)
                        if obj:
                            #print('Old')
                            #print(col_name)
                            col_name = col_name.replace(obj.group(0), '')
                            #print('New')
                            #print(col_name)

                        writer.writerow([col_name])
                        writer.writerow([value])
                        writer.writerow([])

            #df_csv = pd.read_csv(fname, dtype=str)
            #fname = ID + '.xlsx'
            #fname = os.path.join(self.requests_path, folder, folder2, fname)
            #df_csv.to_excel(fname, index=False, header=None)

            break

    def generate_report_for_LTC_resident_questionnaire(self):
        folder = 'Lindsay_apr_05_2023'
        folder2 = 'test'

        regexp_bracket = re.compile('\[(?P<value>.*)\]')
        regexp_dot_number = re.compile('[.]\d+')
        add_another_condition = 'Would you like to add another condition?'
        add_another_medication = 'Would you like to add another medication?'

        fname = 'column_types_for_LTC_resident_questionnaire.xlsx'
        fname = os.path.join(self.requests_path, folder, folder2, fname)
        df_t = pd.read_excel(fname, dtype=str)
        if df_t['Name'].value_counts().gt(1).any():
            vc = df_t['Name'].value_counts()
            s  = vc.gt(1)
            vc = vc[s]
            print(vc)
            raise ValueError('Non unique entries at Name in column type LTC')
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

        #Load the LTC Resident Questionnaire database
        fname = 'LTC_resident_questionnaire.xlsx'
        fname = os.path.join(self.requests_path, folder, folder2, fname)
        df = pd.read_excel(fname, dtype = name_to_type, parse_dates=date_cols)
        #print(df)
        #print(self.parent.print_column_and_datatype(df))
        #return
        df_t.dropna(subset='Section', inplace=True)
        df_t = df_t.sort_values('Section', ascending=True, kind='stable')
        df_t = df_t.groupby('Section')

        SIT = 'Participant Site (2 digits, example: 08)'
        PID = 'Participant ID (7 digits, example: 0123456)'


        for index_m, row_m in df.iterrows():

            site = row_m[SIT]
            pid  = row_m[PID]

            if pd.isnull(site) or pd.isnull(pid):
                continue

            if len(site) != 2:
                raise ValueError('Site length')

            if len(pid) != 7:
                raise ValueError('PID length')

            ID = site + '-' + pid

            fname = ID + '.csv'
            fname = os.path.join(self.requests_path, folder, folder2, fname)

            with open(fname, 'w', newline='') as f:

                writer = csv.writer(f)

                for section, df_g in df_t:
                    writer.writerow([])
                    txt = 'Question #' + section
                    desc= df_g['Description'].iloc[0]
                    writer.writerow([txt, desc])

                    for index_s, row_s in df_g.iterrows():
                        col_name = row_s['Name']
                        value    = row_m[col_name]
                        dtype    = row_s['Type']

                        if pd.isnull(value):
                            continue

                        if dtype == 'date':
                            value = value.strftime('%d-%b-%Y')


                        #Ethnicity
                        #Adverse effects after dose X
                        #Chronic Conditions
                        if section in ['03','30.1','30.2','30.3','30.4','19']:
                            if value == 'No':
                                continue
                            else:
                                obj = regexp_bracket.search(col_name)
                                if obj:
                                    col_name = obj.group('value')
                                    writer.writerow([col_name, value])
                                    writer.writerow([])
                                    continue
                                else:
                                    raise ValueError('Error for bracket')

                        #What is your current height?
                        #What is your current weight?
                        if section in ['07.a','07.b']:
                            obj = regexp_bracket.search(col_name)
                            if obj:
                                col_name = obj.group('value')
                                writer.writerow([col_name, value])
                                writer.writerow([])
                                continue
                            else:
                                raise ValueError('Error for bracket')

                        #Please enter the name of the condition
                        #Would you like to add another condition?
                        #Would you like to add another medication?
                        if add_another_condition in col_name:
                            continue
                        if add_another_medication in col_name:
                            continue
                        obj = regexp_dot_number.search(col_name)
                        if obj:
                            #print('Old')
                            #print(col_name)
                            col_name = col_name.replace(obj.group(0), '')
                            #print('New')
                            #print(col_name)

                        writer.writerow([col_name])
                        writer.writerow([value])
                        writer.writerow([])


            break

    def generate_report_for_one_pager(self):
        folder = 'Lindsay_apr_05_2023'
        folder2 = 'test'

        regexp_bracket = re.compile('\[(?P<value>.*)\]')
        regexp_dot_number = re.compile('[.]\d+')
        add_another_condition = 'Would you like to add another condition?'
        add_another_medication = 'Would you like to add another medication?'

        fname = 'column_types_for_the_one_pager.xlsx'
        fname = os.path.join(self.requests_path, folder, folder2, fname)
        df_t = pd.read_excel(fname, dtype=str)
        if df_t['Name'].value_counts().gt(1).any():
            vc = df_t['Name'].value_counts()
            s  = vc.gt(1)
            vc = vc[s]
            print(vc)
            raise ValueError('Non unique entries at Name in column type LTC')
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

        #Load the one pager
        fname = 'one_pager.xlsx'
        fname = os.path.join(self.requests_path, folder, folder2, fname)
        df = pd.read_excel(fname, dtype = name_to_type, parse_dates=date_cols)
        #print(df)
        #print(self.parent.print_column_and_datatype(df))
        #return
        df_t.dropna(subset='Section', inplace=True)
        df_t = df_t.sort_values('Section', ascending=True, kind='stable')
        df_t = df_t.groupby('Section')

        SIT = 'Participant Site (2 digits, example: 08)'
        PID = 'Participant ID (7 digits, example: 0123456)'


        for index_m, row_m in df.iterrows():

            site = row_m[SIT]
            pid  = row_m[PID]

            if pd.isnull(site) or pd.isnull(pid):
                continue

            if len(site) != 2:
                raise ValueError('Site length')

            if len(pid) != 7:
                raise ValueError('PID length')

            ID = site + '-' + pid

            fname = ID + '.csv'
            fname = os.path.join(self.requests_path, folder, folder2, fname)

            with open(fname, 'w', newline='') as f:

                writer = csv.writer(f)

                for section, df_g in df_t:
                    writer.writerow([])
                    txt = 'Question #' + section
                    desc= df_g['Description'].iloc[0]
                    writer.writerow([txt, desc])

                    for index_s, row_s in df_g.iterrows():
                        col_name = row_s['Name']
                        value    = row_m[col_name]
                        dtype    = row_s['Type']

                        if pd.isnull(value):
                            continue

                        if dtype == 'date':
                            value = value.strftime('%d-%b-%Y')


                        #Adverse effects after dose 5
                        if section == '02':
                            if value == 'No':
                                continue
                            else:
                                obj = regexp_bracket.search(col_name)
                                if obj:
                                    col_name = obj.group('value')
                                    writer.writerow([col_name, value])
                                    writer.writerow([])
                                    continue
                                else:
                                    raise ValueError('Error in adverse effects')



                        writer.writerow([col_name])
                        writer.writerow([value])
                        writer.writerow([])


            break

    def investigate_o_plus_oo_sheraton(self):
        folder= 'Sheraton_mar_23_2023'
        fname = 'single_o.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df_so = pd.read_excel(fname)

        fname = 'double_o.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df_do = pd.read_excel(fname)

        def get_site(txt):
            return txt[:2]

        df_so['Site'] = df_so['ID'].apply(get_site)
        vc_so = df_so['Site'].value_counts()

        df_do['Site'] = df_do['ID'].apply(get_site)
        vc_do = df_do['Site'].value_counts()

        SO = 'Single omicron'
        DO = 'Double omicron'

        df_m = pd.merge(vc_so,vc_do, how='outer', left_index=True, right_index=True)
        df_m = df_m.fillna(0)
        df_m = df_m.reset_index()
        df_m = df_m.rename(columns={'index':'Site', 'Site_x':SO, 'Site_y':DO})
        df_m = pd.melt(df_m, id_vars=['Site'],
                value_vars=[SO, DO],
                var_name='Omicron count',
                value_name = '# of participants')
        print(df_m)
        fig, ax = plt.subplots()
        sns.barplot(ax = ax,
                x='Site', y='# of participants', hue='Omicron count', data=df_m)

        fname = 'o_oo_site_dist.png'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        fig.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close('all')

        folder= 'Sheraton_mar_23_2023'
        fname = 'LTC003_list.xlsx'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df_t = pd.read_excel(fname)
        print(df_t)
        s = df_t['OmicronCount'].notnull()
        df_t = df_t.loc[s]
        fig, ax = plt.subplots()
        sns.histplot(ax = ax,
                x='TimeFromInfection',
                hue='OmicronCount',
                multiple='stack',
                data=df_t)
        fname = 'o_oo_time_from_inf_dist.png'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        fig.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close('all')

    def investigate_single_o_sheraton(self):
        #List of those participants that had a single
        #omicron infection (before BA.5).
        fname = 'single_o.xlsx'
        folder= 'Sheraton_mar_23_2023'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        df_so = pd.read_excel(fname)
        def get_site(txt):
            return txt[:2]
        df_so['Site'] = df_so['ID'].apply(get_site)
        vc = df_so['Site'].value_counts()
        fig, ax = plt.subplots()
        vc.plot(kind='barh', ax = ax)
        ax.set_xlabel('# of participants')
        ax.set_ylabel('Site')
        n = vc.sum()
        txt = 'Single Omicron (BA.1,BA.2) infection'
        txt += ' (n=' + str(n) + ')'
        ax.set_title(txt)

        fname = 'single_o_dist.png'
        fname = os.path.join(self.parent.requests_path, folder, fname)
        fig.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close('all')
