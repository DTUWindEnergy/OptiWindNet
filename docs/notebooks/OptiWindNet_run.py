
from optiwindnet.api import WindFarmNetwork, Heuristic
import numpy as np

cables = [(None, 3, 206), (None, 5, 287), (None, 7, 406)]

substations = np.array([[696, 1063],], dtype=float)
turbines = np.array(
    [[1940, 279], [1920, 703], [1475, 696], [1839, 1250],
     [1277, 1296], [442, 1359], [737, 435], [1060, 26],
     [522, 176], [87, 35], [184, 417], [71, 878]],
    dtype=float
)
border = np.array( # vertices oriented counter-clockwise
    [[1951, 200], [1951, 1383], [386, 1383], [650, 708], [624, 678],
     [4, 1036], [4, 3], [1152, 3], [917, 819], [957, 854]],
    dtype=float)
# 'obstacles' is an optional location attribute
obstacles = [
    # - vertices oriented clockwise for each obstacle polygon
    # - obstacles must be strictly inside the extents polygon
    # - undefined behavior if obstacles and extents borders overlap
    # first obstacle
    np.array([[1540, 920], [1600, 940], [1600, 1150], [1400, 1200]]),
    # [second obstacle] ...
]

# initialize the Heuristic router
router = Heuristic(solver='EW') # default is EW

# create wfn from coordinates
wfn = WindFarmNetwork(turbines=turbines, substations=substations, border=border, obstacles=obstacles, cables=cables,router=router)
#wfn.plot_L()
#wfn.plot_A()

edges_array = wfn.optimize()
# print('from_nodes:', from_nodes)
# print('to_nodes:', to_nodes)
# print('lengths:', lengths)
# print('loads:', loads)
# print('reverses:', reverses)
# print('cable_types:', cable_types)
# print('costs:', costs)
# router should return array tree
#wfn.plot_L()
#wfn.plot_A()
#wfn.plot_G_tentative()
wfn.plot()
print(edges_array['src'])
type(edges_array)
print(edges_array.dtype.names)


wfn.gradient()
