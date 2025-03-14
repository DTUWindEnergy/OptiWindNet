.. OptiWindNet documentation master file

Welcome to OptiWindNet
===========================================

OptiWindNet is an open-sourced and Python-based wind farm electrical network simulation tool developed at DTU 

What Can OptiWindNet Do?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main objective of OptiWindNet is to optimize the electrical network of wind farms.

For installation instructions, please see the :ref:`Installation Guide <installation>`.

Source code repository and issue tracker:
    https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet

License:
    MIT_

.. _MIT: https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/blob/master/LICENSE

Getting Started
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
OptiWindNet is equipped with many capabilities that can range from basic to complex. For new users, the :ref:`Quickstart </notebooks/Quickstart.ipynb>` section shows how to set up and perform some basic operations in OptiWindNet.


How to Cite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Version 1.0.0

`Mauricio Souza de Alencar, Amir Arasteh and Mikkel Friis-Møller. (2025, March).
OptiWindNet 1.0.0: An open-source wind farm electrical network optimization tool. DTU Wind, Technical University of Denmark.`

.. code-block:: python

	@article{
    	    optiwindnet1.0.0_2025,
    	    title={OptiWindNet 1.0.0: An open-source wind farm electrical network optimization tool},
    	    author={Mauricio Souza de Alencar, Amir Arasteh and Mikkel Friis-Møller},
    	    url="https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet",
    	    publisher={DTU Wind, Technical University of Denmark},
    	    year={2025},
    	    month={3}
	    }
..

    .. toctree::
        :maxdepth: 1
	:caption: Essentials

        installation
        notebooks/Quickstart

    .. toctree::
        :maxdepth: 2
	:caption: Features
	:titlesonly:

	notebooks/01-data_input
	notebooks/02-included-locations
	notebooks/03-example-Taylor-2023
	notebooks/04-IEA_Wind_740-10-MW_Reference_Offshore_Wind_Plants
	notebooks/21-MILP_ortools_example
	notebooks/22-MILP_gurobi_example
	notebooks/23-MILP_cplex_example
	notebooks/24-MILP_highs_example
	notebooks/25-MILP_scip_example
	notebooks/26-MILP_cbc_example



    .. toctree::
        :maxdepth: 1
	:caption: API reference

