# NERUON wrapper for Python

Clone repo:

~~~
git clone https://github.com/78furu/neuron_wrapper.git
~~~

### Installation

#### Other packages
It might require other packages like:
- Neuron (sudo apt-install neuron-dev)
- Python3 neuron (pip install neuron)
- OpenMPI (sudo apt install openmpi-bin)

In order to use the basic programs written by Aberra et al., the mechanisms should be run for the NEURON. 

Linux/macOS:
~~~bash
cd neuron_wrapper/
nrnivmodl mechanisms
~~~


Windows:
~~~bash
cd neuron_wrapper/mechanisms
mknrdll 
mv nrnmerch.dll ../nrnmerch.dll
cd ..
~~~

### Usage:
1. check parameters in params.txt (or any specified .p pickle obj conating the dict)
	be careful for filepaths (run_dir must be the same as the mechanisms and cells directory)

2. Run
~~~bash
python3 neuron_wrapper.py params.txt
~~~
