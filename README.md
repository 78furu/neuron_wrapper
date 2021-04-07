# NERUON wrapper for Python

Clone repo:

~~~
git clone https://github.com/78furu/neuron_wrapper.git
~~~

### Installation

In order to use the basic programs written by Aberra et al., the mechanisms should be run. 

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
~~~
python3 neuron_wrapper.py params.txt
~~~
