
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

wfn = WindFarmNetwork(turbines=turbines, substations=substations, border=border, obstacles=obstacles, cables=cables)

router = Heuristic(wfn=wfn, solver='EW') # default is EW
router()
grad_wt, grad_ss = router.gradient()
#print('gradients_wt\n', grad_wt)
#print('gradients_ss\n', grad_ss)
print(wfn.cost())


substations = np.array([[695, 1060],], dtype=float)
router(turbines=turbines, substations=substations)
print(wfn.cost())
#print(wfn.G.graph['VertexC'][-substations.shape[0]:, :])
grad_wt, grad_ss = router.gradient()
#print('gradients_wt\n', grad_wt)
print('gradients_ss\n', grad_ss)
substations = np.array([[695, 1060],], dtype=float)
grad_wt, grad_ss = router.gradient(turbines=turbines, substations=substations)
print('gradients_ss\n', grad_ss)
# print(wfn.get_network())
# a, b = optimzer.gradient()
# print('gradients_wt\n', a)
# print('gradients_ss\n', b)

# substations_new = np.array([[0, 0],], dtype=float)
# wfn.set_coordinates(turbines=turbines, substations=substations_new)

#print('L after optimizer:', wfn.L.graph)
#print('G after optimizer:', wfn.G.graph)

# #wfn = WindFarmNetwork(turbines=turbines, substations=substations, border=border, obstacles=obstacles, cables=cables)
# substations_new = np.array([[0, 0],], dtype=float)
# print('=======================')
# print('here are the VertexC L: ', wfn.L.graph['VertexC'][-1])
# print('here are the VertexC A: ', wfn.A.graph['VertexC'][-1])
# #print('here are the VertexC P: ', wfn.P.graph['VertexC'])
# #print('here are the VertexC S: ', wfn.S.VertexC)
# print('here are the VertexC G: ', wfn.G.graph['VertexC'][-1])

# print('here are the VertexC S: ', wfn.S.graph['T'])
# print('here are the VertexC P: ', wfn.P.graph)
# print('here are the VertexC S: ', wfn.S.graph)
# print("Nodes with attributes S:", wfn.S.nodes(data=True))
# print('=======================')
# wfn.set_coordinates(turbines=turbines, substations=substations_new)
# print('=======================')
# print('here are the VertexC L: ', wfn.L.graph['VertexC'][-1])
# print('here are the VertexC A: ', wfn.A.graph['VertexC'][-1])
# #print('here are the VertexC P: ', wfn.P.graph['VertexC'])
# #print('here are the VertexC S: ', wfn.S.VertexC)
# print('here are the VertexC G: ', wfn.G.graph['VertexC'][-1])
# print('=======================')
# a, b = optimzer.gradient()
# print('gradients_wt\n', a)
# print('gradients_ss\n', b)
