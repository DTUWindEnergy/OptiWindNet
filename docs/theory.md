# Theory

A full analysis of *OptiWindNet*'s methodology can be found in the open-access paper:
- Mauricio Souza de Alencar, Tuhfe Göçmen, Nicolaos A. Cutululis,
_Flexible cable routing framework for wind farm collection system optimization_,
European Journal of Operational Research,
2025, ISSN 0377-2217, <https://doi.org/10.1016/j.ejor.2025.07.069>.

## Formulation

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

## Solution methods

* heuristics based on extensions to the Esau-Williams heuristic for the CMSTP;
* meta-heuristic based on hybrid genetic search ([HGS-CVRP](https://github.com/vidalt/HGS-CVRP) by Thibaut Vidal, described in [his paper](https://doi.org/10.1016/j.cor.2021.105643));
* mathematical optimization using mixed-integer linear programming (MILP) models (using a branch-and-cut solver such as Google's OR-Tools, Coin-OR Branch-and-Cut (CBC), IBM's CPLEX, Gurobi, HiGHS, SCIP).


