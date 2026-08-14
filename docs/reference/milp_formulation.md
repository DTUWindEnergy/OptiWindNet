# MILP Formulation

The exact optimization approach uses a flow-based mixed-integer linear program to select a minimum-length, crossing-free cable topology. The generated formulation below states the common model: binary variables select directed links, continuous flow variables carry turbine power towards a substation, and the constraints enforce capacity, connectivity and planarity.

{py:class}`ModelOptions <optiwindnet.MILP.ModelOptions>` selects the applicable model variant and additional constraints for branched, radial or ringed topology, feeder routing and count, and balanced subtree loads. See [](/routers.md#model-options) for the meaning of those choices and {doc}`/reference/solvers` for the interchangeable solver backends.

The formulation is rendered as native MathML and uses self-hosted, subsetted STIX Two fonts. On narrow displays, scroll the formulation horizontally to see the complete constraints.

````{container} milp-formulation
```{raw} html
:file: ../milp_formulation/problem_formulation.html
```
````

The formulation [LaTeX source and build rules](https://github.com/DTUWindEnergy/OptiWindNet/tree/main/docs/milp_formulation) are distributed with _OptiWindNet_.
