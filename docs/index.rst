.. OptiWindNet documentation master file

About OptiWindNet
=================

OptiWindNet is an electrical network design tool for offshore wind farms developed at DTU.
The package offers a framework to obtain optimal or near-optimal cable routes for a given turbine layout within the cable-laying boundaries. It provides high-level access to heuristic, meta-heuristic and mathematical optimization approaches to the problem.

The tool is distributed as an open-source Python package that is suitable for use within an interactive Python session (e.g. Jupyter notebook). Alternatively, OptiWindNet's API can be invoked directly from another application.

Source code repository:
    https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet

Issue tracker:
    https://github.com/DTUWindEnergy/OptiWindNet/issues

Jupyter notebooks used in this manual:
    https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/tree/main/docs/notebooks

License:
    MIT_

.. _MIT: https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/blob/master/LICENSE

What can OptiWindNet do?
------------------------

* Optimize the network of array cables;
* Route the cables so as to avoid crossings;
* Assign cable types and calculate network costs;
* Use different optimization approaches according to the preferred time/quality trade-off;
* Employ user-provided models and objective functions within the mathematical optimization approach.

Integration with TOPFARM_ is under development.

.. _TOPFARM: https://topfarm.pages.windenergy.dtu.dk/TopFarm2/


Problem Overview
----------------

The design of the collection system of an offshore wind farm can be formulated as a graph problem.
The vertices of the problem represent positions of the power plant's components and are assumed given.
The solution for the problem is a set of edges that represent electrical connections between the components, along with the route along which to lay the electrical cables.

For the solution to be useful, it must meet the constraints that ensure sound electrical circuits and the feasibility of the network:

* circuits can only branch inside a wind turbine, if at all;
* cables cannot cross each other;
* cable routes must fall inside the allowed area, avoiding obstacles within it;
* the maximum current capacity of the cable must be respected.

This problem has similarities with two classic operations research problems:

* The capacitated minimum spanning tree problem (CMSTP);
* The open and capacitated vehicle routing problem (OCVRP);

Neither of the classic formulations consider route crossings, which is the main achievement of OptiWindNet. Whether the approach is via the CMSTP or via the OCVRP depends on the viability of branching the circuits on turbines. OptiWindNet can produce both branched and radial (non-branching) networks.

Approaches
^^^^^^^^^^

* heuristics based on extensions to the Esau-Williams heuristic for the CMSTP;
* meta-heuristic based on hybrid genetic search (`implemented elsewhere <https://github.com/vidalt/HGS-CVRP>`_ by Thibaut Vidal, described in `his paper <https://doi.org/10.1016/j.cor.2021.105643>`_);
* mathematical optimization using mixed-integer linear programming (MILP) models (using a branch-and-cut solver such as Google's OR-Tools, Coin-OR Branch-and-Cut (CBC), IBM's CPLEX, Gurobi, HiGHS, SCIP).


Getting Started
---------------

Ensure that the :ref:`Requirements` are met and follow the :ref:`Installation` instructions. Then check the :doc:`Quickstart <notebooks/Quickstart>` to begin using OptiWindNet.


How to Cite
-----------
Version 1.0.0

`Mauricio Souza de Alencar, Amir Arasteh and Mikkel Friis-Møller. (2025, March).
OptiWindNet 1.0.0: An open-source wind farm electrical network optimization tool. DTU Wind, Technical University of Denmark.`

.. code-block:: bib

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
        :maxdepth: 2
	:caption: Essentials

        setup
        notebooks/Quickstart

    .. toctree::
	:maxdepth: 2
	:caption: Features

	features

    .. toctree::
	:maxdepth: 2
	:caption: Paper

	paper_experiments

    .. toctree::
        :maxdepth: 2
	:caption: API reference

