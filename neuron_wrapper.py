import os, sys
import subprocess
import numpy as np
import pickle 



def generate_waveform(stim_amps, DEL, DUR, resolution = 10):
    
    stim_times = []
    
    for _ in stim_amps:
        stim_times.append(DUR)
    total_length = np.zeros(int((DEL + sum(stim_times))*resolution))
    tax = np.linspace(0, DEL + sum(stim_times), len(total_length))
    i = 0
    c = 1
    waveform = []
    while i < len(total_length):
        t = tax[i]
        if t < DEL:
            waveform.append(0)
        elif DEL +  sum(stim_times[:c-1]) <= t < DEL + sum(stim_times[:c]):
            waveform.append(stim_amps[c-1])
        elif i == (DEL + sum(stim_times[:c]))*resolution:
            waveform.append(stim_amps[c-1])
            c += 1
        else:
            waveform.append(stim_amps[c-1])
        i +=1
    return waveform, tax

# def create_params_file(cell_id, theta_0, phi_0, DEL, DUR, AMP, tstop
def create_params_file(cell_id, theta_0, phi_0, DEL, DUR, AMP, tstop, dt, 
                       fname_to_save_data, fname_to_save_params, stim_mode = 2 ):
    f = open(fname_to_save_params, "w")
    stg = f"""
cell_id = {cell_id}
theta = {theta_0} 
phi_0 = {phi_0}
v_init = {v_inits.get(cell_id, -73)}

stim_mode = {stim_mode} // 1 = ICMS, 2 = uniform E-field

DEL = {DEL}  // ms - delay to first phase
DUR = {DUR} // ms - duration of first phase
AMP = {AMP}  // µA (stim_mode = 1) or V/m (stim_mode = 2) - amplitude of first phase

tstop = {tstop}
dt = {dt}
strdef fname_to_save
fname_to_save = "{fname_to_save_data}"
"""
    f.write(stg)
    f.close()

# def create_stim_file(stim_amps, fname_to_save, AMP = 1,  openmode = 'w',
def create_stim_file(stim_amps, fname_to_save, AMP = 1,  openmode = 'w',
                     return_stim_wfv = True):
    stim_amps =np.array( stim_amps)*AMP
    stim_waveform_vector = []
    stim_waveform_vector.append(0)
    stim_waveform_vector.append(0)

    for i in stim_amps:
        stim_waveform_vector.append(i)
        stim_waveform_vector.append(i)

    stim_waveform_vector.append(0)
    stim_waveform_vector.append(0)
    
    vector_creation_string=f"""
objref stim_amp
stim_amp = new Vector()
stim_amp.resize({len(stim_waveform_vector)})
stim_amp.fill(0)
"""
    vector_filling_string = """"""
    
    for i in range(2, len(stim_waveform_vector)-2):
        vector_filling_string += f"stim_amp.x[{i}] = {stim_waveform_vector[i]}\n "
    
    stim_vector_string = vector_creation_string + vector_filling_string 
    f = open(fname_to_save, openmode)
    f.write(stim_vector_string)
    f.close()
    if return_stim_wfv:
        return stim_waveform_vector

# def create_time_file(stim_waveform_vector, DEL, DUR, fname_to_save, openmode = 'a')
def create_time_file(stim_waveform_vector, DEL, DUR, fname_to_save, openmode = 'a'):

    time_creation_string=f"""
objref stim_time
stim_time = new Vector()
stim_time.resize({len(stim_waveform_vector)})
stim_time.fill(0)
"""

    time_filling_string = """"""

    for i in range(1, len(stim_waveform_vector)-1):
        time_filling_string += f"stim_time.x[{i}] = {DEL + DUR*((i-1)//2)}\n "
    time_filling_string += f"stim_time.x[{len(stim_waveform_vector)-1}] = {DEL + DUR*((i-1)//2) +1}\n"

    time_vector_string = time_creation_string + time_filling_string 

    f = open(fname_to_save, openmode)
    f.write(time_vector_string)
    f.close()

# def create_thetas_file(thetas, DEL, DUR, fname_to_save, openmode = "a")
def create_thetas_file(thetas, DEL, DUR, fname_to_save, openmode = "a"):
    thetas_creation_string=f"""
objref thetas
thetas = new Vector()
thetas.resize({len(thetas)})
thetas.fill(0)
"""

    thetas_filling_string = """"""

    for i in range(len(thetas)):
        thetas_filling_string += f"thetas.x[{i}] = {thetas[i]}\n "

    thetas_vector_string = thetas_creation_string + thetas_filling_string # + vector_print_string
    
    thetas_time_borders = [DEL + c*DUR for c in range(len(thetas)+1)]
    
    thetas_time_creation_string=f"""
objref section_time
section_time = new Vector()
section_time.resize({len(thetas_time_borders)})
section_time.fill(0)
"""

    thetas_time_filling_string = """"""

    for i in range(len(thetas_time_borders)):
        thetas_time_filling_string += f"section_time.x[{i}] = {thetas_time_borders[i]}\n "

    thetas_time_vector_string = thetas_time_creation_string + thetas_time_filling_string # + vector_print_string
    
    f = open(fname_to_save, openmode)
    f.write(thetas_vector_string)
    f.write(thetas_time_vector_string)
    f.close()
    


# def create_thetas_file(thetas, DEL, DUR, fname_to_save, openmode = "a")
def create_phis_file(phis, fname_to_save, openmode = "a"):
    phis_creation_string=f"""
objref phis
phis = new Vector()
phis.resize({len(phis)})
phis.fill(0)
"""

    phis_filling_string = """"""

    for i in range(len(phis)):
        phis_filling_string += f"phis.x[{i}] = {phis[i]}\n "

    phis_vector_string = phis_creation_string + phis_filling_string # + vector_print_string
    
    
    f = open(fname_to_save, openmode)
    f.write(phis_vector_string)
    f.close()


def create_all_params(stim_amps, theta_0, thetas, phi_0, phis, DEL, DUR, AMP, cell_id, 
                     tstop, dt, fname_to_save_data, fname_to_save_params,
                     nevezektan, compartment, part, fname_to_save_rec, stim_mode = 2, **kwargs):
    
    create_params_file(cell_id = cell_id, theta_0=theta_0, phi_0=phi_0, DEL=DEL, 
                       DUR = DUR, AMP=AMP, tstop=tstop, dt = dt,
                       fname_to_save_data = fname_to_save_data, 
                       fname_to_save_params = fname_to_save_params,
                       stim_mode = stim_mode )
    stim_waveform_vector = create_stim_file(stim_amps = stim_amps, 
                                            fname_to_save = fname_to_save_params,
                                            AMP = AMP,  openmode = 'a',
                                            return_stim_wfv = True)
    
    create_time_file(stim_waveform_vector = stim_waveform_vector, DEL = DEL, 
                     DUR = DUR, fname_to_save = fname_to_save_params, openmode = 'a')
    
    create_thetas_file(thetas = thetas, DEL = DEL, DUR = DUR, 
                       fname_to_save = fname_to_save_params, openmode = "a")

    create_phis_file(phis = phis, fname_to_save = fname_to_save_params, openmode = "a" )

    create_recording_file(nevezektan = nevezektan, compartment = compartment, part = part,
                    save_to = fname_to_save_rec,)


def create_project_details(params):
    project_path = params["project_path"]
    project_name = params["project_name"]
    try:
        os.mkdir(project_path)
    except FileExistsError:
        pass
    pickle.dump( params, open( project_path + project_name + '_params.p', "wb" ) )        


def spiking_times(data):
    time_ax, soma_volt = data[:, 0], data[:, 1]
    spike_place = [ time_ax[v]  
        for v in range(len(soma_volt)-1) if (soma_volt[v]*soma_volt[v+1]<0 and soma_volt[v+1] >=0)
    ]
    return spike_place

def get_spike_counts(resp, params):
    spikes_per_theta = {t:[] for t in params["thetas"]}
    theta_limits = {params["DEL"] + params["DUR"]*c : t  for c, t in enumerate(params["thetas"])}
    spiking_time = spiking_times(resp)

    for spike in spiking_time:
        lt = [th for ti, th in theta_limits.items() if ti<=spike]
        spikes_per_theta[lt[-1]].append(spike)

    spike_counts = {k: sum([1 for _ in v ]) for k, v in spikes_per_theta.items()}
    return spike_counts


def create_recording_file(nevezektan, compartment="soma", part = 0.5, save_to = None):
    is_list_in_list = False if isinstance(nevezektan[0], str) else True
    num_of_labels = sum([len(nevezek) for nevezek in nevezektan]) if is_list_in_list else \
                                 len(nevezektan)
    calling = "objref rect, recv" 
    vectoring = 'rect = new Vector()\nrecv = new Vector()\n'
    recording = f'rect.record(&t)\nrecv.record(&cell.{compartment}.v({part}))\n'
    printing= """
objref savdata
savdata = new File()
savdata.wopen(fname_to_save)"""

    printing += f"""
savdata.printf("# time {compartment}.voltage"""

    for nev in [nev for nev in nevezektan if nev[-1] != ']']:
        printing += f" {compartment}.{nev}" 

    printing +="""\\n")
for i=0,rect.size()-1 {
"""

    printing_inner= 'savdata.printf("%g %g '


    for _ in [nev for nev in nevezektan if nev[-1] != ']']:
        printing_inner += "%g "
    printing_inner += '\\n\", rect.x(i), recv.x(i),'

    if is_list_in_list:
        for c0, nevezek in enumerate(nevezektan):
            calling += "\nobjref"
            for c, nev in enumerate(nevezek):
                if c!= len(nevezek)-1:
                    calling += f" rec{nev},"
                    printing_inner += f"rec{nev}.x(i), "

                else:
                    calling += f" rec{nev}"
                    if c0 != len(nevezektan)-1:
                        printing_inner += f"rec{nev}.x(i), "
                    else:
                        printing_inner += f"rec{nev}.x(i))"

                vectoring += f"rec{nev} = new Vector()\n"
                recording += f"rec{nev}.record(&cell.{compartment}.{nev}({part}))\n"
    else:
        for c, nev in enumerate(nevezektan):
            if nev[-1] == "]":
                continue
            calling += "\nobjref"
            if c!= len(nevezektan)-1:
                
                calling += f" rec{nev}"
                printing_inner += f"rec{nev}.x(i), "

            else:
                calling += f" rec{nev}"
                printing_inner += f"rec{nev}.x(i))"

            vectoring += f"rec{nev} = new Vector()\n"
            recording += f"rec{nev}.record(&cell.{compartment}.{nev}({part}))\n"
    printing += printing_inner + """
}

savdata.close()
"""
    
    final = calling + '\n\n' + vectoring + recording + '\nrun()\n\n' + '\n' + printing
    
    if save_to is not None:
        f = open(save_to, 'w')
        f.write(final)
        f.close()
    return final


amps = np.linspace(100, 8000, 15)
neurons = list(range(1, 26))

# params
stim_amps = []

# stim_amps_fill

stim_amps.append(1)
stim_amps.append(1)
stim_amps.append(1)
stim_amps.append(1)
stim_amps.append(1)
stim_amps.append(1)
stim_amps.append(1)
stim_amps.append(1)
stim_amps.append(1)
stim_amps.append(1)

#stim_amps = [np.sin(i) for i in np.linspace(0, 3.14, 10)]

theta_0 = 90 

#thetas = [90, 180, 0, 180, 0, 180, 0, 180, 0, 180]
thetas = list(np.linspace(0, 180,10 ) )
phi_0 = 0

phis = [phi_0 for _ in thetas]
DEL = 1  # ms - delay to first phase
DUR = 50 # ms - duration of first phase
AMP = 200  # µA (stim_mode = 1) or V/m (stim_mode = 2) - amplitude of first phase

cell_id = 15

stim_mode = 2 # 1 = ICMS, 2 = uniform E-field


tstop = 220
dt = 0.025

run_dir = '.'
fname_to_save_data = 'proba.dat'
fname_to_save_params = run_dir + "/params.hoc"
fname_to_save_rec = run_dir + '/record_params.hoc'
v_inits = {1: -74.739,
         2: -72.728,
         3: -71.803,
         4: -73.324,
         5: -73.952,
         7: -72.991,
         10: -72.557,
         11: -70.741,
         13: -70.066,
         14: -71.464,
         15: -71.672,
         19: -71.275,
         20: -71.163,
         21: -73.324,
         22: -72.937,
         24: -73.024}


nevezektan_soma = ["cai", "cao", "cm", "decay_CaDynamics_E2", "depth_CaDynamics_E2", "diam",
         "dica_dv_", "dik_dv_", "dina_dv_", "e_extracellular", "e_pas", "eca",  "ek" ,"ena",
         "es_xtra", "ex_xtra", "gCa_LVAst_Ca_LVAst", "gCa_LVAstbar_Ca_LVAst", 
         "gCabar_Ca", "gIm_Im", "gImbar_Im", "gK_Pst_K_Pst", "gK_Pstbar_K_Pst", "gK_Tst_K_Tst",
         "gK_Tstbar_K_Tst", "gNaTs2_t_NaTs2_t",  "gNaTs2_tbar_NaTs2_t", "gNap_Et2_Nap_Et2",  
         "gNap_Et2bar_Nap_Et2", "gSK_E2_SK_E2", "gSK_E2bar_SK_E2", "gSKv3_1_SKv3_1", 
         "gSKv3_1bar_SKv3_1", "g_pas", "gamma_CaDynamics_E2", "h_Ca", "h_Ca_LVAst", "h_K_Pst",
         "h_K_Tst", "h_NaTs2_t", "h_Nap_Et2", "i_cap", "i_membrane", "i_pas", "ica", "ica_Ca",
         "ica_Ca_LVAst", "ik", "ik_Im", "ik_K_Pst", "ik_K_Tst", "ik_SK_E2", "ik_SKv3_1", 
         "ina", "ina_NaTs2_t", "ina_Nap_Et2", "ki", "ko", "m_Ca", "m_Ca_LVAst", "m_Im", 
         "m_K_Pst", "m_K_Tst", "m_NaTs2_t", "m_Nap_Et2", "m_SKv3_1", "minCai_CaDynamics_E2",
         "nai", "nao", "order_xtra", "type_xtra", "v", "vext[0]", "vext[1]", "x_xtra", "xc[0]", 
         "xc[1]", "xg[0]", "xg[1]", "xraxial[0]", "xraxial[1]", "y_xtra", "z_SK_E2", "z_xtra" 
         ]

nevezektan_soma_old = [
    ['eca', 'ek', 'ena'],
    ['dica_dv_', 'dik_dv_', 'dina_dv_'],
    ['e_extracellular', 'e_pas', 'g_pas'], 
    ['gCa_Ca', 'gIm_Im', 'gK_Pst_K_Pst', 'gNaTs2_t_NaTs2_t'],
    ['h_Ca', 'h_K_Pst', 'h_NaTs2_t'],
    ['i_cap', 'i_membrane', 'i_pas'],
    ['ica', 'ik', 'ina'], 
    ['m_Ca', 'm_K_Pst', 'm_NaTs2_t'],
    ['cai', 'cao', 'ki', 'ko', 'nai', 'nao']
]
nevezektan_axon = [
    ['eca', 'ek', 'ena'],
    ['dica_dv_', 'dik_dv_', 'dina_dv_'],
    ['e_extracellular', 'e_pas', 'g_pas'], 
    ['gCa_Ca', 'gIm_Im', 'gK_Pst_K_Pst', 'gNaTa_t_NaTa_t'],
    ['h_Ca', 'h_K_Pst', 'h_NaTa_t'],
    ['i_cap', 'i_membrane', 'i_pas'],
    ['ica', 'ik', 'ina'], 
    ['m_Ca', 'm_K_Pst', 'm_NaTa_t'],
    ['cai', 'cao', 'ki', 'ko', 'nai', 'nao']
]

nevezektan_dend = [
    ['ek', 'ena'],
    ['dik_dv_', 'dina_dv_'],
    ['e_extracellular', 'e_pas', 'g_pas'], 
    ['gIm_Im', 'gK_Pst_K_Pst', 'gNaTs2_t_NaTs2_t'],
    ['h_K_Pst', 'h_NaTs2_t'],
    ['i_membrane', 'i_pas'],
    ['ik', 'ina'], 
    ['m_K_Pst', 'm_NaTs2_t'],
    ['ki', 'ko', 'nai', 'nao']
]

nevezektan_dict = {"soma" : nevezektan_soma, "axon": nevezektan_axon, "dend": nevezektan_dend}

params = {}

params["stim_amps"] = stim_amps.copy()
params["theta_0"]  = theta_0
params["thetas"] = thetas.copy()
params["phi_0"] = phi_0
params["phis"] = phis
params["DEL"], params["DUR"], params["AMP"] = DEL, DUR, AMP
params["cell_id"] = cell_id
params["stim_mode"] = stim_mode
params["tstop"] = tstop
params['dt'] = dt
params["fname_to_save_data"] = fname_to_save_data
params["fname_to_save_params"] = fname_to_save_params
params['nevezektan'] = nevezektan_soma
params['compartment'] = 'soma'
params['part'] = 0.5
params['fname_to_save_rec'] = fname_to_save_rec
params['project_path'] = run_dir 
params['project_name'] = 'TEMP'
params['fname_to_save_data'] = params['project_path'] + params['project_name'] +  '.dat'

def run(params):
    create_all_params(**params)
    create_project_details(params)
    orig_pwd = os.getcwd()
    run_dir = params['run_dir']
    os.chdir(run_dir)
    subprocess.run(["nrniv", "init.hoc"])
    os.chdir(orig_pwd)



if __name__ == "__main__":
    try:
        del os.environ['DISPLAY']
    except:
        pass
    if len(sys.argv) < 2:
        run(params= params)
    elif len(sys.argv) == 2:
        if sys.argv[1][-2:] == '.p':
            params = pickle.load(open(sys.argv[-1], "rb"))
            run(params= params)
        elif sys.argv[1][-4:] == '.txt':
            exec(open(sys.argv[1]).read())
            run(params= params)


