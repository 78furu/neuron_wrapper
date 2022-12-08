# neuron utils for python
def import_stuff(path, file):
    import sys, os
    os.chdir(path)
    global h
    global ms
    global mV
    from neuron import h, rxd
    h.load_file('import3d.hoc')
    from neuron.units import ms, mV
    h.load_file(file)



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.cluster import KMeans
from collections import Counter


DEFAULT_PATH = '/home/furu/Documents/neuron/L5bPyrCellHayEtAl2011/NEURON/'
DEFAULT_FILE = 'simulationcode/stim.hoc'


SECTION_TYPES = ['soma', 'dend', 'apic', 'axon', 'Myelin', 'Node', 'Unmyelin',  'xScale', 'yScale', 'zScale']

SECTION_COLORS = {'soma' : 'k', 'axon' : 'tab:orange', 'apic' : 'tab:blue', 
                        'dend':'tab:green', 'Myelin':'tab:grey', 'Node':'tab:red', 'Unmyelin': 'tab:orange'}



def run(v_init = -70, tstop = 50):
    t = h.Vector().record(h._ref_t)                     # Time stamp vector

    h.load_file('stdrun.hoc')
    h.finitialize(v_init* mV)
    h.continuerun(tstop * ms)
    return list(t)

def get_sectype(section, sectypes = SECTION_TYPES):
    for sectype in sectypes:
        if section.hname().find(sectype) != -1:
            return sectype

def create_Vectors(section, cut = 0.5,):
    vectors = {}
    for mech in section(cut):
        for m in mech:
            if m.name() in ['ihcn_Ih', 'ik', 'ina', 'ica']:
                sec = section
                num_of_3d_points = len(sec.psection()['morphology']['pts3d'])-1
                r0 = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
                r1 = np.array([sec.x3d(num_of_3d_points), sec.y3d(num_of_3d_points), sec.z3d(num_of_3d_points)])
                dist_between_ends = np.linalg.norm(r1-r0)
                pts3d = sec.psection()['morphology']['pts3d']

                rel_dists_from_latest = [0]

                for segcunt in range(len(pts3d)-1):
                    p0 = np.array(pts3d[segcunt])[:-1]
                    p1 = np.array(pts3d[segcunt + 1])[:-1]
                    rel_dists_from_latest.append(np.linalg.norm(p1-p0)/sec.L)
                rel_dists_from_start = np.cumsum(rel_dists_from_latest)
                l = []
                for cut in rel_dists_from_start:
                    exec(f'l.append(h.Vector().record(section(cut)._ref_{m.name()}))') 
                vectors[m.name()] = l
            else:
                exec(f'vectors[m.name()] = h.Vector().record(section(cut)._ref_{m.name()})')
            
    return vectors

def create_cellinfo_database(all_secs, sectypes = SECTION_TYPES):
    all_sec_for_dv = []
    type_counter = {sectype :0 for sectype in sectypes}
    for c, sec in enumerate(all_secs):
        sectype = get_sectype(sec)
        if sectype.find('Scale') != -1:
            print(sectype)
            print('skip')
            continue
        if sectype == "soma":
            num_of_3d_points = len(sec.psection()['morphology']['pts3d'])-1
            middle_of_soma = ((sec.x3d(num_of_3d_points)+sec.x3d(0))/2, (sec.y3d(num_of_3d_points)+sec.y3d(0))/2, (sec.z3d(num_of_3d_points)+sec.z3d(0))/2)
        try:
            num_of_3d_points = len(sec.psection()['morphology']['pts3d'])-1
            r0 = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
            r1 = np.array([sec.x3d(num_of_3d_points), sec.y3d(num_of_3d_points), sec.z3d(num_of_3d_points)])
            dist_between_ends = np.linalg.norm(r1-r0)
            pts3d = sec.psection()['morphology']['pts3d']

            rel_dists_from_latest = [0]

            for segcunt in range(len(pts3d)-1):
                p0 = np.array(pts3d[segcunt])[:-1]
                p1 = np.array(pts3d[segcunt + 1])[:-1]
                rel_dists_from_latest.append(np.linalg.norm(p1-p0)/sec.L)
            rel_dists_from_start = np.cumsum(rel_dists_from_latest)
            props = (c, sec, sectype, type_counter[sectype], sec.x3d(0), sec.y3d(0), sec.z3d(0), \
                     sec.x3d(num_of_3d_points), sec.y3d(num_of_3d_points), sec.z3d(num_of_3d_points), \
                     sec.psection()['morphology']['pts3d'],\
                     sec.nseg,\
                     [ sec(i).diam   for i in np.linspace(0.001, 1,sec.nseg, endpoint=False)],\
                     [ sec(i).area()   for i in np.linspace(0.001, 1,sec.nseg, endpoint=False)], \
                     np.linalg.norm((sec.x3d(num_of_3d_points)-middle_of_soma[0], sec.y3d(num_of_3d_points-1)-middle_of_soma[1], sec.z3d(num_of_3d_points)-middle_of_soma[2])), not bool(sec.children()),\
                     sec.L, \
                     sec.psection()['Ra'], [],\
                     [h.Vector().record(sec(cut)._ref_v) for cut in rel_dists_from_start],\
                    [h.Vector().record(sec(cut)._ref_i_cap) for cut in rel_dists_from_start], 
                    [h.Vector().record(sec(cut)._ref_i_pas) for cut in rel_dists_from_start],)  
        except:
            try:
                print(f'Error w {c}, {sectype}')
                props = (c, sec, sectype, type_counter[sectype], np.nan, np.nan, np.nan, \
                 np.nan, np.nan, np.nan, \
                 sec.psection()['morphology']['pts3d'],\
                 sec.nseg,\
                 [ sec(i).diam   for i in np.linspace(0.001, 1,sec.nseg, endpoint=False)],\
                 [ sec(i).area()   for i in np.linspace(0.001, 1,sec.nseg, endpoint=False)], \
                 np.nan, not bool(sec.children()),\
                 sec.L, \
                 sec.psection()['Ra'],[],\
                 [h.Vector().record(sec(0.5)._ref_v)],  \
                [h.Vector().record(sec(0.5)._ref_i_cap)], 
                np.nan) 
            except AttributeError:
                print(f'Error w {c}, {sectype}')
                props = (c, sec, sectype, type_counter[sectype], np.nan, np.nan, np.nan, \
                 np.nan, np.nan, np.nan, \
                 sec.psection()['morphology']['pts3d'],\
                 sec.nseg,\
                 [ sec(i).diam   for i in np.linspace(0.001, 1,sec.nseg, endpoint=False)],\
                 [ sec(i).area()   for i in np.linspace(0.001, 1,sec.nseg, endpoint=False)], \
                 np.nan, not bool(sec.children()),\
                 sec.L, \
                 sec.psection()['Ra'],[],\
                 [h.Vector().record(sec(0.5)._ref_v)],  \
                [h.Vector().record(sec(0.5)._ref_i_cap)], 
                [h.Vector().record(sec(0.5)._ref_i_pas)]) 
        all_sec_for_dv.append(props)
        type_counter[sectype] += 1
    return pd.DataFrame(all_sec_for_dv, columns=['id', 'sec', 'type', 'type_id', 'x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'pts3d', 'nseg', 'diams_per_segment','areas_per_segment',  'dist_of_end_from_soma_center', 'distal', 'L', 'Ra', 'inserted_stim', 'v', 'i_cap', 'i_pas'])


def create_recordings_database(all_secs):
    all_mechs = {} 
    all_mechnames = set()
    for c, section in enumerate(all_secs):
        mechs = create_Vectors(section)
        all_mechs[c] = mechs
        all_mechnames = all_mechnames.union(set(mechs.keys()))


    df = pd.DataFrame([], index= list(all_mechs.keys()), columns=all_mechnames)

    for c, f in all_mechs.items():
        for n, V in f.items():
            df.iloc[c][n] = V
    return df


def get_cell_database(all_secs):
    df_info = create_cellinfo_database(all_secs)
    df_rec = create_recordings_database(all_secs)
    df = pd.merge(df_info, df_rec, left_index=True, right_index=True)
    childrens = []
    parents = []
    for d in range(len(df)):
        children = []
        for c in df.iloc[d].sec.children():
            df_ = df[df.sec == c]
            children.append((df_.type.values[0], df_.type_id.values[0], df_.distal.values[0]))
        childrens.append(children)
        
        
        try:
            s = df.iloc[d].sec.trueparentseg().sec
            df_ = df[df.sec == s]
            parents.append((df_.type.values[0], df_.type_id.values[0]))
        except AttributeError:
            parents.append(None)
    df['children'] = childrens
    df['parent'] = parents

    cols = list(df)
    cols.insert(4, cols.pop(cols.index('children')))
    cols.insert(5, cols.pop(cols.index('parent')))
    df = df.loc[:, cols]

    return df

    

def choose_section_for_targeting(df, kmeans = None, distance = 'closer', sectype = 'apic', num = 10):
    if sectype == 'dend':
        chosen_section_ids = tuple(sorted(np.random.choice(df[df.type == 'dend']['id'], size = (num))))
        return chosen_section_ids
    elif sectype == 'apic' and kmeans is not None:
        if distance == 'closer':
            dist_marker = 0
        elif distance == 'further':
            dist_marker = 1
        else:
            raise ValueError(f'Distance can only be  _closer_ (def) or _further_ but not {distance}.')
        chosen_section_ids = set()
        while len(chosen_section_ids)<num:
            cid = np.random.choice(df[df.type == 'apic']['id'])
            if (kmeans.predict(np.array([df.iloc[cid]['dist_of_end_from_soma_center']]).reshape(1, -1)) == 1 and df.iloc[cid]['distal'] == True)\
            or (kmeans.predict(np.array([df.iloc[cid]['dist_of_end_from_soma_center']]).reshape(1, -1)) == 0):
                chosen_section_ids.update([cid])
        return tuple(sorted(chosen_section_ids))
    else:
        raise ValueError('For apic, kmeans is required, for dend it is not, and no other sectypes are included.')

def insert_stim(cid, df, stimtype = 'alpha', cut = 1):
    if stimtype == 'alpha':
        f = h.AlphaSynapse(df.iloc[cid]['sec'](cut))
        f.onset = 10
        f.tau = .1
        f.gmax = 1
        f.e = 40
        df.iloc[cid]['inserted_stim'].append(f)
    else:
        raise TypeError(f'stimtype {stimtype} is not implemented, try _alpha_.')

def insert_input(cell, segtype, which = 'all'):
    stims = []
    for c in range(len(cell.dend)):
        s = h.IClamp(cell.dend[c](0.5))
        s.delay = .5
        s.dur = 5
        s.amp = 100
        stims.append(s)
        #stims[c].e = 40
    

def insert_to_apic_dend(df, plot = False, insert_to_ids = None, distance = 'closer'):

    dend_ends = np.array(df[(df.type == 'dend')][['dist_of_end_from_soma_center']].sort_values('dist_of_end_from_soma_center').values)
    apic_ends = np.array(df[(df.type == 'apic')][['dist_of_end_from_soma_center']].sort_values('dist_of_end_from_soma_center').values)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(apic_ends)
    if plot:

        fig, ax = plt.subplots(1,1, figsize=(6,3))
        ax.plot(dend_ends, '.', label = 'dend ends')
        ax.plot(apic_ends, '.', label = 'apic ends')
        ax.plot([Counter(kmeans.labels_)[0]/2, Counter(kmeans.labels_)[0] + Counter(kmeans.labels_)[1]/2 ],
               kmeans.cluster_centers_, 's', markersize = 10, label = 'Cluster centers')
        ax.set_xlabel('distance from soma center')
        ax.set_ylabel('count')
        ax.legend()
    #fig.savefig('apic_dend_distances_kmeans.png', dpi = 150)

    if insert_to_ids is None:
        chosen_section_ids = choose_section_for_targeting(df, kmeans, distance=distance)
        for cid in chosen_section_ids:
            insert_stim(cid, df)
    else:
        chosen_section_ids = insert_to_ids
        for cid in chosen_section_ids:
            insert_stim(cid, df)

def format_df(df):
    ty = type(df.iloc[1].v[0])

    df_ = df.copy()

    for col in df_.columns:
        
        df_[col] = df_[col].apply(lambda x: x.as_numpy() if (type(x) == ty and hasattr(x, 'hname') and x.hname().find('Vector') !=-1) else x)

        df_[col] = df_[col].apply(lambda x: [i.to_python() for i in x] if (type(x) == list and len(x) >0 and type(x[0]) == ty and hasattr(x[0], 'hname') and x[0].hname().find('Vector') !=-1) else x)
        df_[col] = df_[col].apply(lambda x: None  if (type(x) == list and len(x) ==0 ) else x)
        df_[col] = df_[col].apply(lambda x: True if (type(x)==list and 
                                                     len(x) == 1 and 
                                                     type(x[0]) ==ty and 
                                                     x[0].hname().find('AlphaSynapse') !=-1) else x)
    return df_

class NeuronDataBase(object):
    """docstring for NeuronDataBase"""
    def __init__(self, path, file, insert_to_ids = None, distance = 'closer'):
        super(NeuronDataBase, self).__init__()
        self.path = path
        self.file = file
        self.insert_to_ids = insert_to_ids
        self.distance = distance

        import_stuff(self.path, self.file)

        self.df = df = get_cell_database(list(h.allsec()))

    def insert_stim(self):
        insert_to_apic_dend(self.df, insert_to_ids = self.insert_to_ids, distance = self.distance)
    def run(self, tstop, v_init):
        self.t = run(tstop = tstop, v_init = v_init)

