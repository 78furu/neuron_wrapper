# --------------------------------------
# | Neuron wrapper by Kristof Furuglyas|
# --------------------------------------
# This code snippet is designed to give a Python3 interpreter to the NEURON's hoc snytax, 
# and to create a possibility to simulate single neurons with various extracellular 
# fields having different directions and amplitudes.
# 
# 2021, Neunos Ltd.

import os, sys
import subprocess
import numpy as np
import pickle 



def generate_waveform(stim_amps, DEL, DUR, resolution = 10):
    """
    Generating a waveform for later visualization. This function is not used for the 
    core run. 

        stim_amps:          list of floats, for (relative) amplitudes
        DEL:                float, delay, awaiting time (in ms) before 1st stim
        DUR:                float, duration of one section
        resolution:         int, num of points taken in one stimulatory segment

    Returns the list of floats (waveform itself) and the timeaxis.
    """
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
def create_params_file(cell_id, theta_0, phi_0, DEL, DUR, AMP, tstop, dt, v_inits, 
                       fname_to_save_data, fname_to_save_params, stim_mode = 2 ):
    """
    Creating the parameters file to determine the run specifications from given args.
    This function creates a file written in hoc language. Rewrites _fname_to_save_params_
    file totally.

        cell_id:            int [1, 25], see Aberra et al. for specifications
        theta_0:            float [degrees], azimuthal angle WHEN NO STIMULUS IS PRESENT
        phi_0:              float [degrees], polar angle WHEN NO STIMULUS IS PRESENT
        DEL:                float [ms], time before first segment of stim starts
        DUR:                float [ms], length in time of one segment of stim
        AMP:                float [mV/mm], overall stimulus amplitude with which all of
                                _stim_amps_ are going to be multiplied with
        tstop:              float [ms], total stimulation time
        dt:                 float [ms], temporal inverse resolution; time between two steps
        v_inits:            dict (cell_id, v_init) [mV], starting (resting) voltage of cells
        fname_to_save_data: str, filename to save output data
        fname_to_save_params: str, filename to save parameter hoc file (what neuron is 
                                going to load in -- recommended/default "params.hoc")
        stim_mode:          int [1 or 2], 1 = ICMS, 2 = uniform E-field, latter implemented

    Returns nothing, creates hoc file under _fname_to_save_params_ name.   
    """
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
    """
    Creating stimulus amplitude hoc text. Recommended to use with openmode = a, 
    to append to fname_to_save_params (see create_params_file) file. 

        stim_amps:          list (floats), containing the relative amplitudes compared
                                to _AMP_ parameter. Those are multiplied together.
        fname_to_save:      str, filename to print created hoc text
        AMP:                float [mV/mm], overall stimulus amplitude with which all of
                                _stim_amps_ are going to be multiplied with, def = 1
        openmode:           'w' (def) or 'a', opening mode for file opening in Python3 
                                (w: write; a: append) 
        return_stim_wfv:    bool, whether to return stimulus waveform (in hoc lang),
                                def = True
    
    Return nothing, places text to fname_to_save.
    """

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
    """
    Creating time vector hoc text. Recommended to use with openmode = 'a', 
    to append to fname_to_save_params (see create_params_file) file.

        stim_waveform_vector:   list or np.array, waveform in hoc format but 
                                    on python lang
        DEL:                    float [ms], time before first segment of stim starts
        DUR:                    float [ms], length in time of one segment of stim   
        fname_to_save:          str, filename to print created hoc text
        openmode:               'w' (def) or 'a', opening mode for file opening in
                                    Python3 (w: write; a: append) 
    
    Return nothing, places text to fname_to_save.
    """

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
    """
    Creating thetas vector hoc text. Thetas vector defines the azimuth angle 
    of ec. field in every segment. Recommended to use with openmode = 'a', 
    to append to fname_to_save_params (see create_params_file) file.

        thetas:             list (floats) [degrees], containing the azimuth
                                angles in each segment
        DEL:                float [ms], time before first segment of stim starts
        DUR:                float [ms], length in time of one segment of stim   
        fname_to_save:      str, filename to print created hoc text
        openmode:           'a' (def) or 'w', opening mode for file opening in
                                Python3 (w: write; a: append) 
    
    Return nothing, places text to fname_to_save.
    """
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
    """
    Creating phis vector hoc text. Phis vector defines the polar angle of 
    ec. field in every segment. Recommended to use with openmode = 'a', 
    to append to fname_to_save_params (see create_params_file) file.

        phis:               list (floats) [degrees], containing the polar
                                angles in each segment
        DEL:                float [ms], time before first segment of stim starts
        DUR:                float [ms], length in time of one segment of stim   
        fname_to_save:      str, filename to print created hoc text
        openmode:           'a' (def) or 'w', opening mode for file opening in
                                Python3 (w: write; a: append) 
    
    Return nothing, places text to fname_to_save.
    """    
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

def create_recording_file(nevezektan, compartment="soma", part = 0.5,
                            save_to = None, save_all_params = True):
    """
    Creating the recording hoc file. This file determines what segments and 
    parameters to be recorded during run. Furthermore, it contains the core 
    run() function that starts and supervises the simulation. If _save_all_
    params_ is True, then nevezektan must be a dict and saves all possible 
    params for all possible compartments, and other params are ignored (see
    create_recording_file_all).

        nevezektan:         list of lists of strings OR dict (compartment,
                                list of...) containg the names of what to
                                rec (nomenclature)
        compartment:        str, "soma", "axon" or "dend" (or "apic" where
                                applicable) determining which section to rec,
                                def = "soma"
        part:               float [0,1], on given compartment where to
                                record, def = 0.5
        save_to:            str, save file under name, if None (def) then 
                                only returns the text
        save_all_params:    bool, if True, dict is needed for nevezektan,
                                runs for all compartments, otherwise other
                                params are ignored
    
    Returns created hoc text and if save_to is not None then saves under that 
    filename.
    """
    if save_all_params:
        create_recording_file_all(nevezektan_dict=nevezektan, save_to=save_to)
        return 0
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


def create_recording_file_all(nevezektan_dict, save_to = None):
    """
    Creating recording hoc file. This functions requires a dict as a nevezektan,
    iterates over all compartments and saves the recording text to save_to.
    Returns the text, too, and if save_to is not None, saves under that name.
    
    """
    calling = "objref rect" 
    vectoring = 'rect = new Vector()\n'
    recording = f'rect.record(&t)\n'
    printing= """
objref savdata
savdata = new File()
savdata.wopen(fname_to_save)"""

    printing += f"""
savdata.printf("time"""
    
    printing_inner= 'savdata.printf("%g '

    for compartment, (part, nevezektan) in nevezektan_dict.items():
        for nev in [nev for nev in nevezektan if nev[-1] != ']']:
            printing += f" {compartment}.{nev}" 
            printing_inner += "%g "
    printing_inner += '\\n\", rect.x(i), '
    printing +="""\\n")
for i=0,rect.size()-1 {
"""


    for c0, (compartment, (part, nevezektan)) in enumerate(nevezektan_dict.items()):

        for c, nev in enumerate(nevezektan):
            if nev[-1] == "]":
                continue
            calling += "\nobjref"
            if c!= len(nevezektan)-1:
                
                calling += f" rec_{compartment}_{nev}"
                printing_inner += f"rec_{compartment}_{nev}.x(i), "

            else:
                if c0 == len(list(nevezektan_dict.keys()))-1:
                    calling += f" rec_{compartment}_{nev}"
                    printing_inner += f"rec_{compartment}_{nev}.x(i))"
                else:
                    calling += f" rec_{compartment}_{nev}"
                    printing_inner += f"rec_{compartment}_{nev}.x(i), "

            vectoring += f"rec_{compartment}_{nev} = new Vector()\n"
            recording += f"rec_{compartment}_{nev}.record(&cell.{compartment}.{nev}({part}))\n"
    
    

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


def create_all_params(stim_amps, theta_0, thetas, phi_0, phis, DEL, DUR, AMP, cell_id, 
                     tstop, dt, v_inits, fname_to_save_data, fname_to_save_params,
                     nevezektan, compartment, part, fname_to_save_rec, save_all_params, 
                     stim_mode = 2, **kwargs):
    """
    Wrapper function to run all the functions needed to create the hoc files. 

        stim_amps:          list of floats, for (relative) amplitudes
        theta_0:            float [degrees], azimuthal angle WHEN NO STIMULUS IS PRESENT
        thetas:             list (floats) [degrees], containing the azimuth
                                angles in each segment
        phi_0:              float [degrees], polar angle WHEN NO STIMULUS IS PRESENT
        phis:               list (floats) [degrees], containing the polar
                                angles in each segment
        DEL:                float [ms], time before first segment of stim starts
        DUR:                float [ms], length in time of one segment of stim
        AMP:                float [mV/mm], overall stimulus amplitude with which all of
                                _stim_amps_ are going to be multiplied with
        cell_id:            int [1, 25], see Aberra et al. for specifications
        tstop:              float [ms], total stimulation time
        dt:                 float [ms], temporal inverse resolution; time between two steps
        v_inits:            dict (cell_id, v_init) [mV], starting (resting) voltage of cells
        fname_to_save_data: str, filename to save output data
        fname_to_save_params: str, filename to save parameter hoc file (what neuron is 
                                going to load in -- recommended/default "params.hoc")
        nevezektan:         list of lists of strings OR dict (compartment,
                                list of...) containg the names of what to
                                rec (nomenclature)
        compartment:        str, "soma", "axon" or "dend" (or "apic" where
                                applicable) determining which section to rec,
                                def = "soma"
        part:               float [0,1], on given compartment where to
                                record, def = 0.5
        fname_to_save_rec:  str, save file under name, if None (def) then 
                                only returns the text
        save_all_params:    bool, if True, dict is needed for nevezektan,
                                runs for all compartments, otherwise other
                                params are ignored
        stim_mode:          int [1 or 2], 1 = ICMS, 2 = uniform E-field, latter 
                                implemented 

    Keyword arguments are NOT passed anywhere, it is there so throws no error for
    unknown parameters.

    Returns nothing explicitly, prints to files with respect to parameters.  
    
    """
    
    create_params_file(cell_id = cell_id, theta_0=theta_0, phi_0=phi_0, DEL=DEL, 
                       DUR = DUR, AMP=AMP, tstop=tstop, dt = dt, v_inits=v_inits,
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
                    save_to = fname_to_save_rec, save_all_params=save_all_params)


def create_project_details(params):
    """
    Creates project path folder and pickles params.
        params:                 dict,
            project_path (key): str, path of the project
            project_name (key): str, name of the project
    
    Returns nothing, pickles out all parameters into a .p file.
    """
    project_path = params["project_path"]
    project_name = params["project_name"]
    try:
        os.mkdir(project_path)
    except FileExistsError:
        pass
    pickle.dump( params, open( project_path + project_name + '_params.p', "wb" ) )        


def spiking_times(data):
    """
    Finds spiking times in data. Data has a shape [2, *any*], first column  
    is time (in ms) second is voltage (mV). Returns list of spiking times.

    Note: there is a spike in the nth time point if the nth voltage is negative
    and the (n+1)th voltage is positive. Yes, it assumes appropriate data res.
    """
    time_ax, soma_volt = data[:, 0], data[:, 1]
    spike_place = [ time_ax[v]  
        for v in range(len(soma_volt)-1) if (soma_volt[v]*soma_volt[v+1]<0 and \
                                        soma_volt[v+1] >=0)
    ]
    return spike_place

def get_spike_counts(resp, params):
    """
    Get spike counts for a total time series but with respect to different
    segments. 

        resp:               np.array (shape: [2, *any*]), data for 
                                spiking_times function
        params:             dict, containg all the parameters for the run
    
    Returns a dict (thetas, num of spikes). 
    """
    spikes_per_theta = {t:[] for t in params["thetas"]}
    theta_limits = {params["DEL"] + params["DUR"]*c : t  for c, t in enumerate(params["thetas"])}
    spiking_time = spiking_times(resp)

    for spike in spiking_time:
        lt = [th for ti, th in theta_limits.items() if ti<=spike]
        spikes_per_theta[lt[-1]].append(spike)

    spike_counts = {k: sum([1 for _ in v ]) for k, v in spikes_per_theta.items()}
    return spike_counts



def run(params):
    """
    Running all the neccessary fuctions before the execution, the execute run.
    Changing directory to appropriate path, run, change back.

        params:             str, filename ending in .txt which must be 
                                a python script having "params" as a dict

    Returns nothing. 
    """
    if isinstance(params,str):
        if params[-4:] == '.txt':
            exec(open(params).read())
    create_all_params(**params)
    create_project_details(params)
    orig_pwd = os.getcwd()
    run_dir = params['run_dir']
    os.chdir(run_dir)
    subprocess.run(["nrniv", "-nogui",  "-NSTACK", "100000", "-NFRAME", "20000", "init.hoc", "-c", "quit()"])
    os.chdir(orig_pwd)



if __name__ == "__main__":
    try:
        del os.environ['DISPLAY']
    except:
        pass
    if len(sys.argv) < 2:
        sys.exit("Prespecified parameters is deprecated, please use extarnal .p or .txt file for params")
        # run(params= params)
    elif len(sys.argv) == 2:
        if sys.argv[1][-2:] == '.p':
            params = pickle.load(open(sys.argv[-1], "rb"))
            run(params= params)
        elif sys.argv[1][-4:] == '.txt':
            exec(open(sys.argv[1]).read())
            run(params= params)
    else:
        sys.exit("Usage: \npython3 neuron_wrapper.py params.txt")


