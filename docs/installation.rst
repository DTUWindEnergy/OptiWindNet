.. _installation:

Installation Guide
===========================


Pre-Installation
----------------------------
Before you can install the software, you must first have a working Python distribution with a package manager. For all platforms we recommend that you download and install Anaconda - a professional grade, full-blown scientific Python distribution.

To set up Anaconda, you should:

    * Download and install Anaconda (Python 3.x version, 64 bit installer is recommended) from https://www.continuum.io/downloads
    
    * Update the root Anaconda environment (type in a terminal): 
        
        ``>> conda update --all``
    
    * Activate the Anaconda root environment in a terminal as follows: 
        
        ``>> activate``

It is recommended to create a new environment to install OptiWindNet if you have other Python programs. This ensures that the dependencies for the different packages do not conflict with one another. In the command prompt, create and active an environment with::

   git clone https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet.git
   cd OptiWindNet
   conda env create -f environment.yml
   activate optiwindnet_env


Simple Installation
----------------------------

OptiWindNet’s base code is open-sourced and freely available on `GitLab 
<https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet>`_ (MIT license).

* Install from PyPi.org (official releases)::
  
    pip install optiwindnet

* Install from GitLab  (includes any recent updates)::
  
    pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet.git
        


Developer Installation
-------------------------------

We highly recommend developers to install OptiWindNet into the environment created previously. The commands to clone and install OptiWindNet with developer options including dependencies required to run the tests into the current active environment in an Anaconda Prompt are as follows::

   git clone https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet.git
   cd OptiWindNet
   pip install -e .[test]