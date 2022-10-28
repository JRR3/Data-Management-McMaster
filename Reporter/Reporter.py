#JRR @ McMaster University
#Update: 10-Oct-2022
import numpy as np
import pandas as pd
pd.options.plotting.backend = 'plotly'
import plotly.express as pxp
import os
import re
import networkx as nx
import matplotlib as mpl
mpl.rcParams['figure.dpi']=300
import matplotlib.pyplot as plt
import datetime
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


    def find_short_jumps(self):
        #This function was updated on 13-Oct-2022
        folder = 'Jenna_oct_6_2022'
        fname = 'table.xlsx'
        fname = os.path.join(self.requests_path, folder, fname)
        df = pd.read_excel(fname)
        print('# rows df:', len(df))
        rel_columns = ['Anti.inflamm.and.DMARDs',
                       'TNF.inhib',
                       'IL6.inhib',
                       'JAK.inhib',
                       'Costim.Inhib']
        df_med = df[rel_columns].copy()
        M = df_med.values
        M = M.astype(int)
        vec_to_count = {}
        #We are going to count how many unique rows we have.
        for row in M:
            s = str(row)
            v = vec_to_count.get(s,0)
            vec_to_count[s] = v + 1
        df_unique_rows = pd.Series(vec_to_count).to_frame()
        df_unique_rows = df_unique_rows.reset_index()
        df_unique_rows = df_unique_rows.rename(columns={'index':'Combination',
                                                        0:'Counts'})
        index_to_label = {k:chr(x) for k,x in enumerate(range(65,91))}
        vec_to_label = {}
        for k,(key,count) in enumerate(vec_to_count.items()):
            vec_to_label[key] = index_to_label[k]
        df_unique_rows = df_unique_rows.rename(index=index_to_label)

        print(df_unique_rows)
        n_unique_rows = len(vec_to_count)

        df_unique_rows = df_med.drop_duplicates()
        M = df_unique_rows.values

        G = nx.Graph()

        for k, row1 in enumerate(M):
            for j in range(k, n_unique_rows):
                row2 = M[j]
                dist = np.linalg.norm(row1-row2, 1)
                if dist < 1.5:
                    #Connected
                    s1 = str(row1)
                    s2 = str(row2)
                    l1 = index_to_label[k]
                    w1 = vec_to_count[s1]
                    l2 = index_to_label[j]
                    w2 = vec_to_count[s2]
                    if k == j:
                        continue
                        G.add_edge(l1,l2)
                    else:
                        l1 += ',' + str(w1)
                        l2 += ',' + str(w2)
                        G.add_edge(l1,l2)
                        G.add_edge(l1,l2)
        print(G)
        pos = nx.spring_layout(G)
        pos = nx.circular_layout(G)
        nx.draw_networkx_nodes(G,
                               pos,
                               cmap=plt.get_cmap('Wistia'),
                               node_color = list(vec_to_count.values()),
                               node_size = 800)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edge_color='b')
        #edge_labels=dict([((u,v),w) for u,v,w in G.edges.data('weight')])
        #nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
        plot_name = 'graph.png'
        fname = os.path.join(self.dpath, plot_name)
        plt.savefig(fname)
        return


        unique_rows = df_med.drop_duplicates()
        print(unique_rows)
        print('# unique rows:', len(unique_rows))
        print('# unique rows:', n_unique_rows)
        return
        print(unique_rows.to_string(index=False))
        individuals = df['Individual'].unique()
        print('# of individuals:', df['Individual'].nunique())
        for x in individuals:
            selector = df['Individual'] == x
            rows = df.loc[selector, rel_columns]
            n_rows = len(rows)
            for med in rel_columns:
                if rows[med].nunique() == 1:
                    pass
                else:
                    print('==========')
                    print(x)
                    print(rows)


    def plot_serology_slopes_from_selection(self):
        fpure = 'infection_dates_slope.xlsx'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        df    = pd.read_excel(fname, sheet_name = 'selection')
        fig, ax = plt.subplots()
        n_rows = len(df)
        mk_size = 4
        for k, (index, row) in enumerate(df.iterrows()):
            di = row['Infection date']
            d1 = row['S: Date before']
            d2 = row['S: Date after']
            dt_1 = row['S: Days before']
            IgG1 = row['S: Nuc-IgG-100 before']
            IgG2 = row['S: Nuc-IgG-100 after']
            IgA1 = row['S: Nuc-IgA-100 before']
            IgA2 = row['S: Nuc-IgA-100 after']
            IgG_slope = row['S: Slope IgG']
            IgA_slope = row['S: Slope IgA']
            IgG_mid = IgG1 + dt_1 * IgG_slope
            IgA_mid = IgA1 + dt_1 * IgA_slope
            ax.plot([d1,d2], [IgG1, IgG2], 'b-')
            ax.plot([di], [IgG_mid], 'ko', markersize=mk_size)
            ax.plot([d1,d2], [IgA1, IgA2], 'r-')
            ax.plot([di], [IgA_mid], 'ko', markersize=mk_size)
            if k == n_rows - 1:
                ax.plot([d1,d2], [IgG1, IgG2], 'b-', label='IgG')
                ax.plot([d1,d2], [IgA1, IgA2], 'r-', label='IgA')
        plt.legend(loc='best')
        ax.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b-%y"))
        ax.set_ylabel('OD')
        plt.xticks(rotation=45)
        fpure = 'slope_plot_selection.png'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        fig.savefig(fname)

        plt.close('all')
        fig, ax = plt.subplots()
        G_h, G_bins = np.histogram(df['S: Slope IgG'])
        A_h, A_bins = np.histogram(df['S: Slope IgA'], bins=G_bins)
        G_width = (G_bins[1] - G_bins[0])
        A_width = (A_bins[1] - A_bins[0])
        width = np.max((G_width, A_width))
        left = np.min((G_bins[0], A_bins[0]))
        right = np.max((G_bins[-1], A_bins[-1]))
        if left < 0 and right > 0:
            n = int(np.ceil(np.abs(left) / width))
            left_partition = np.flip(-np.linspace(0,n*width,n+1))
            n = int(np.ceil((right) / width))
            right_partition = np.linspace(width,n*width,n+1)
            partition = np.concatenate((left_partition, right_partition))
        else:
            n_left = np.ceil(np.abs(left) / width)
            if right < 0:
                n_right = np.floor(np.abs(right) / width)
            else:
                n_right = np.ceil(np.abs(right) / width)
            a = np.sign(left)*n_left*width
            b = np.sign(right)*n_right*width
            partition = np.arange(a,b,width)

        width /= 3
        fs = 12
        G_h, _ = np.histogram(df['S: Slope IgG'], bins=partition)
        A_h, _ = np.histogram(df['S: Slope IgA'], bins=partition)
        #Generate labels
        base = partition[0]
        L = []
        for x in partition[1:]:
            #print(base, x)
            interval = '({:.1E},{:.1E})'.format(base, x)
            interval = interval.replace('+0','+')
            interval = interval.replace('-0','-')
            interval = interval.replace('-.0','0')
            interval = interval.replace('+.0','0')
            #print(interval)
            base = x
            L.append(interval)
            #print('---')
        ax.bar(L, G_h, width=0.5,  facecolor='blue', label='IgG')
        ax.bar(L, A_h,  width=0.25, facecolor='red', label='IgA')
        ax.set_xlabel('Slope')
        ax.set_ylabel('Count')
        #ticks = ax.xaxis.get_ticklocs()
        #print(ticks)
        #ticklabels = ax.xaxis.get_ticklabels()
        #print(ticklabels)
        #ax.xaxis.set_ticks(ticks)
        #ax.xaxis.set_ticklabels(L)
        plt.xticks(fontsize=fs, rotation=90)
        #plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        plt.legend(loc='best', fontsize=fs)
        plt.tight_layout()
        fpure = 'hist_plot_Igs.png'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        fig.savefig(fname)

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

    def plot_serology_slope_vs_bins_after_infection(self):
        fpure = 'infection_dates_slope.xlsx'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        df    = pd.read_excel(fname)
        df['S: Nearest IgG'] = df['S: Nuc-IgG-100 before'].where(
            df['S: Days before'] < df['S: Days after'],
            df['S: Nuc-IgG-100 after'])
        df['S: Nearest IgA'] = df['S: Nuc-IgA-100 before'].where(
            df['S: Days before'] < df['S: Days after'],
            df['S: Nuc-IgA-100 after'])
        selection  = df['Method'] == 'PCR'
        selection &= df['# infections'] == 1
        selection &= df['S: Has slopes?'] == True
        print('Available data (negative slopes included):')
        print(selection.value_counts())
        df_s = df.loc[selection, :]
        #=====Bins IgG
        plt.close('all')
        fig, ax = plt.subplots()
        t_max = df_s['S: Days after'].max()
        print(f'{t_max=}')
        width = 30
        n_bins = int(np.ceil(t_max / width))
        print(f'{n_bins=}')
        dc = {}
        base = 0
        bin_labels = []
        index_to_label = {}
        for k in range(1,n_bins+1):
            txt = f'[{base},{width*k})'
            bin_labels.append(txt)
            base = width*k
            index_to_label[k-1] = txt
            dc[txt] = []
            print(txt)
        for index, row in df_s.iterrows():
            days_after = row['S: Days after']
            IgG_slope = row['S: Slope IgG']
            bin_index = days_after // width
            bin_label = index_to_label[bin_index]
            dc[bin_label].append(IgG_slope)
        ax.boxplot(dc.values())
        ax.set_xticklabels(dc.keys())
        ax.set_ylabel('IgG (Slope)')
        ax.set_xlabel('Days after infection')
        plt.xticks(rotation=60)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.tight_layout()

        fpure = 'Slope_IgG_vs_bin_days_after.png'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        fig.savefig(fname)

        #=====Bins IgA
        plt.close('all')
        fig, ax = plt.subplots()
        t_max = df_s['S: Days after'].max()
        print(f'{t_max=}')
        width = 30
        n_bins = int(np.ceil(t_max / width))
        print(f'{n_bins=}')
        dc = {}
        base = 0
        bin_labels = []
        index_to_label = {}
        for k in range(1,n_bins+1):
            txt = f'[{base},{width*k})'
            bin_labels.append(txt)
            base = width*k
            index_to_label[k-1] = txt
            dc[txt] = []
            print(txt)
        for index, row in df_s.iterrows():
            days_after = row['S: Days after']
            IgA_slope = row['S: Slope IgA']
            bin_index = days_after // width
            bin_label = index_to_label[bin_index]
            dc[bin_label].append(IgA_slope)
        ax.boxplot(dc.values())
        ax.set_xticklabels(dc.keys())
        ax.set_ylabel('IgA (Slope)')
        ax.set_xlabel('Days after infection')
        plt.xticks(rotation=60)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.tight_layout()

        fpure = 'Slope_IgA_vs_bin_days_after.png'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        fig.savefig(fname)


    def plot_serology_slope_vs_days_after_infection(self):
        fpure = 'infection_dates_slope.xlsx'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        df    = pd.read_excel(fname)
        df['S: Nearest IgG'] = df['S: Nuc-IgG-100 before'].where(
            df['S: Days before'] < df['S: Days after'],
            df['S: Nuc-IgG-100 after'])
        df['S: Nearest IgA'] = df['S: Nuc-IgA-100 before'].where(
            df['S: Days before'] < df['S: Days after'],
            df['S: Nuc-IgA-100 after'])
        selection  = df['Method'] == 'PCR'
        selection &= df['# infections'] == 1
        print('Available data (negative slopes included):')
        print(selection.value_counts())
        df_s = df.loc[selection, :]

        #Scatter
        plt.close('all')
        fig, ax = plt.subplots()
        ax.scatter(df_s['S: Days after'],
                   df_s['S: Slope IgG'], color='blue')
        #Regression
        selection = df_s['S: Slope IgG'] > 0
        df_G = df_s.loc[selection, :]
        n_G = len(df_G)
        print(f'{n_G=}')
        x = df_G['S: Days after'].values
        y_G = df_G['S: Slope IgG'].values
        f_G = np.polyfit(x, np.log(y_G), 1)
        k_G = f_G[0]
        c_G = np.exp(f_G[1])
        l_G = -np.log(2) / k_G
        print(f'{k_G=:.2E}')
        print(f'{c_G=:.2E}')
        print(f'{l_G=:.2E}')
        xx = np.linspace(np.min(x), np.max(x), 100)
        yy_G = c_G * np.exp( k_G * xx)
        ax.plot(xx, yy_G,'k-')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_xlabel('Days after infection')
        ax.set_ylabel('IgG slope')
        fpure = 'Slope_IgG_vs_days_after.png'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        fig.savefig(fname)

        #Scatter
        plt.close('all')
        fig, ax = plt.subplots()
        ax.scatter(df_s['S: Days after'], df_s['S: Slope IgA'], color='red')
        #Regression
        selection = df_s['S: Slope IgA'] > 0
        df_A = df_s.loc[selection, :]
        n_A = len(df_A)
        print(f'{n_A=}')
        x = df_A['S: Days after'].values
        y_A = df_A['S: Slope IgA'].values
        f_A = np.polyfit(x, np.log(y_A), 1)
        k_A = f_A[0]
        c_A = np.exp(f_A[1])
        l_A = -np.log(2) / k_A
        print(f'{k_A=:.2E}')
        print(f'{c_A=:.2E}')
        print(f'{l_A=:.2E}')
        xx = np.linspace(np.min(x), np.max(x), 100)
        yy_A = c_A * np.exp( k_A * xx)
        ax.plot(xx, yy_A,'k-')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_xlabel('Days after infection')
        ax.set_ylabel('IgA slope')
        fpure = 'Slope_IgA_vs_days_after.png'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        fig.savefig(fname)


        #=========================Nearest
        print('Nearest')
        #Scatter
        plt.close('all')
        fig, ax = plt.subplots()
        ax.scatter(df_s['S: Days after'],
                   df_s['S: Nearest IgG'], color='blue')
        #Regression
        selection = df_s['S: Nearest IgG'] > 0
        df_G = df_s.loc[selection, :]
        n_G = len(df_G)
        print(f'{n_G=}')
        x = df_G['S: Days after'].values
        y_G = df_G['S: Nearest IgG'].values
        f_G = np.polyfit(x, np.log(y_G), 1)
        k_G = f_G[0]
        c_G = np.exp(f_G[1])
        l_G = -np.log(2) / k_G
        print(f'{k_G=:.2E}')
        print(f'{c_G=:.2E}')
        print(f'{l_G=:.2E}')
        xx = np.linspace(np.min(x), np.max(x), 100)
        yy_G = c_G * np.exp( k_G * xx)
        ax.plot(xx, yy_G,'k-')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_xlabel('Days after infection')
        ax.set_ylabel('Nearest IgG')
        fpure = 'Nearest_IgG_vs_days_after.png'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        fig.savefig(fname)

        #Scatter
        plt.close('all')
        fig, ax = plt.subplots()
        ax.scatter(df_s['S: Days after'], df_s['S: Nearest IgA'], color='red')
        #Regression
        selection = df_s['S: Nearest IgA'] > 0
        df_A = df_s.loc[selection, :]
        n_A = len(df_A)
        print(f'{n_A=}')
        x = df_A['S: Days after'].values
        y_A = df_A['S: Nearest IgA'].values
        f_A = np.polyfit(x, np.log(y_A), 1)
        k_A = f_A[0]
        c_A = np.exp(f_A[1])
        l_A = -np.log(2) / k_A
        print(f'{k_A=:.2E}')
        print(f'{c_A=:.2E}')
        print(f'{l_A=:.2E}')
        xx = np.linspace(np.min(x), np.max(x), 100)
        yy_A = c_A * np.exp( k_A * xx)
        ax.plot(xx, yy_A,'k-')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_xlabel('Days after infection')
        ax.set_ylabel('Nearest IgA')
        fpure = 'Nearest_IgA_vs_days_after.png'
        folder= 'Braeden_oct_20_2022'
        fname = os.path.join(self.parent.requests_path, folder, fpure)
        fig.savefig(fname)




