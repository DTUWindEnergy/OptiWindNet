# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import abc
import logging
from dataclasses import asdict, dataclass

from enum import StrEnum, auto
from itertools import chain
from typing import Any, Mapping

import networkx as nx
from makefun import with_signature

from ..interarraylib import G_from_S
from ..pathfinding import PathFinder

_lggr = logging.getLogger(__name__)
error, info = _lggr.error, _lggr.info


def _identifier_from_class_name(c: type) -> str:
    "Convert a camel-case class name to a snake-case identifier"
    s = c.__name__
    return s[0].lower() + ''.join('_' + c.lower() if c.isupper() else c for c in s[1:])


class OWNWarmupFailed(Exception):
    pass


class OWNSolutionNotFound(Exception):
    pass


class Topology(StrEnum):
    "Set the topology of subtrees in the solution."

    RADIAL = auto()
    BRANCHED = auto()
    DEFAULT = BRANCHED


class FeederRoute(StrEnum):
    'If feeder routes must be "straight" or can be detoured ("segmented").'

    STRAIGHT = auto()
    SEGMENTED = auto()
    DEFAULT = SEGMENTED


class FeederLimit(StrEnum):
    'Whether to limit the maximum number of feeders, if set to "specified", additional kwarg "max_feeders" must be given.'

    UNLIMITED = auto()
    SPECIFIED = auto()
    MINIMUM = auto()
    MIN_PLUS1 = auto()
    MIN_PLUS2 = auto()
    MIN_PLUS3 = auto()
    DEFAULT = UNLIMITED


class ModelOptions(dict):
    """Hold options for the modelling of the cable routing problem.

    Use ModelOptions.help() to get the options and their permitted and default
    values. Use ModelOptions() without any parameters to use the defaults.
    """

    hints = {
        _identifier_from_class_name(kind): kind
        for kind in (Topology, FeederRoute, FeederLimit)
    }
    # this has to be kept in sync with make_min_length_model()
    simple = dict(
        balanced=(
            bool,
            False,
            'Whether to enforce balanced subtrees (subtree loads differ at most '
            'by one unit).',
        ),
        max_feeders=(
            int,
            0,
            'Maximum number of feeders (used only if <feeder_limit = "specified">)',
        ),
    )

    @with_signature(
        '__init__(self, *, '
        + ', '.join(
            chain(
                (f'{k}: {v.__name__} = "{v.DEFAULT.value}"' for k, v in hints.items()),
                (
                    f'{name}: {kind.__name__} = {default}'
                    for name, (kind, default, _) in simple.items()
                ),
            )
        )
        + ')'
    )
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, str):
                kwargs[k] = self.hints[k](v)
            else:
                if k not in self.simple:
                    raise ValueError(f'Unknown argument: {k}')

        super().__init__(kwargs)

    @classmethod
    def help(cls):
        for k, v in cls.hints.items():
            print(
                f'{k} in {{'
                + ', '.join(
                    f'"{m}"' for n, m in v.__members__.items() if n != 'DEFAULT'
                )
                + f'}} default: {cls.hints[k].DEFAULT.value}\n'
                f'    {v.__doc__}\n'
            )
        for name, (kind, default, desc) in cls.simple.items():
            print(f'{name} [{kind.__name__}] default: {default}\n    {desc}\n')


@dataclass(slots=True)
class ModelMetadata:
    R: int
    T: int
    capacity: int
    linkset: tuple
    link_: Mapping
    flow_: Mapping
    model_options: dict
    fun_fingerprint: dict[str, str | bytes]
    warmed_by: str = ''


@dataclass(slots=True)
class SolutionInfo:
    runtime: float
    bound: float
    objective: float
    relgap: float
    termination: str


class Solver(abc.ABC):
    "Common interface to multiple MILP solvers"

    name: str
    metadata: ModelMetadata
    solver: Any
    options: dict[str, Any]
    stopping: dict[str, Any]
    solution_info: SolutionInfo
    applied_options: dict[str, Any]

    @abc.abstractmethod
    def _link_val(self, var: Any) -> int | bool:
        "Get the value of a link variable from the current solution."
        pass

    @abc.abstractmethod
    def _flow_val(self, var: Any) -> int:
        "Get the value of a flow variable from the current solution."
        pass

    @abc.abstractmethod
    def set_problem(
        self,
        P: nx.PlanarEmbedding,
        A: nx.Graph,
        capacity: int,
        model_options: ModelOptions,
        warmstart: nx.Graph | None = None,
    ):
        """Define the problem geometry, available edges and tree properties

        Args:
          P: planar embedding of the location
          A: available edges for the location
          capacity: maximum number of terminals in a subtree
          model_options: tree properties - see ModelOptions.help()
          warmstart: initial feasible solution to pass to solver
        """
        pass

    @abc.abstractmethod
    def solve(
        self,
        time_limit: float,
        mip_gap: float,
        options: dict[str, Any] = {},
        verbose: bool = False,
    ) -> SolutionInfo:
        """Run the MILP solver search.

        Args:
          time_limit: maximum time (s) the solver is allowed to run.
          mip_gap: relative difference from incumbent solution to lower bound
            at which the search may be stopped before time_limit is reached.
          options: additional options to pass to solver (see solver manual).

        Returns:
          General information about the solution search (use get_solution() for
            the actual solution).
        """
        pass

    @abc.abstractmethod
    def get_solution(self, A: nx.Graph | None = None) -> tuple[nx.Graph, nx.Graph]:
        """Output solution topology A and routeset G.

        Args:
          A: optionally replace the A given via set_problem() (if normalized A)

        Returns:
          Topology graph S and routeset G.
        """
        pass

    def _make_graph_attributes(self) -> dict[str, Any]:
        metadata, solution_info = self.metadata, self.solution_info
        solver_details = self.applied_options.copy()
        # the method_options dict is extracted by db utility function packmethod()
        method_options = dict(
            solver_name=self.name,
            fun_fingerprint=metadata.fun_fingerprint,
            **self.stopping,
            **metadata.model_options,
        )
        # remaining graph attributes (key=value) are stored in db.RouteSet[].misc
        attr = dict(
            **asdict(solution_info),
            method_options=method_options,
            solver_details=solver_details,
        )
        if 'max_feeders' in method_options:
            solver_details['max_feeders'] = method_options.pop('max_feeders')
        if metadata.warmed_by:
            attr['warmstart'] = metadata.warmed_by
        return attr

    def _topology_from_mip_sol(self):
        """Create a topology graph from the solution to the MILP model.

        Returns:
          Graph topology `S` from the solution.
        """
        metadata = self.metadata
        S = nx.Graph(R=metadata.R, T=metadata.T)
        # ensure roots are added, even if some are not connected
        S.add_nodes_from(range(-metadata.R, 0))
        # Get active links and if flow is reversed (i.e. from small to big)
        rev_from_link = {
            (u, v): u < v
            for (u, v), var in metadata.link_.items()
            if self._link_val(var)
        }
        S.add_weighted_edges_from(
            (
                (u, v, self._flow_val(metadata.flow_[u, v]))
                for (u, v) in rev_from_link.keys()
            ),
            weight='load',
        )
        # set the 'reverse' edge attribute
        nx.set_edge_attributes(S, rev_from_link, name='reverse')
        # propagate loads from edges to nodes
        subtree = -1
        max_load = 0
        for r in range(-metadata.R, 0):
            for u, v in nx.edge_dfs(S, r):
                S.nodes[v]['load'] = S[u][v]['load']
                if u == r:
                    subtree += 1
                S.nodes[v]['subtree'] = subtree
            rootload = 0
            for nbr in S.neighbors(r):
                subtree_load = S.nodes[nbr]['load']
                max_load = max(max_load, subtree_load)
                rootload += subtree_load
            S.nodes[r]['load'] = rootload
        S.graph.update(
            capacity=metadata.capacity,
            max_load=max_load,
            has_loads=True,
            creator='MILP.' + self.name,
            solver_details={},
        )
        return S


class PoolHandler(abc.ABC):
    name: str
    num_solutions: int
    model_options: ModelOptions

    @abc.abstractmethod
    def _objective_at(self, index: int) -> float:
        "Get objective value from solution pool at position `index`"
        pass

    @abc.abstractmethod
    def _topology_from_mip_pool(self) -> nx.Graph:
        "Build topology from the pool solution at the last requested position"
        pass

    def _investigate_pool(
        self, P: nx.PlanarEmbedding, A: nx.Graph
    ) -> tuple[nx.Graph, nx.Graph]:
        """Go through the solver's solutions checking which has the shortest length
        after applying the detours with PathFinder."""
        Λ = float('inf')
        branched = self.model_options['topology'] is Topology.BRANCHED
        num_solutions = self.num_solutions
        info(f'Solution pool has {num_solutions} solutions.')
        for i in range(num_solutions):
            λ = self._objective_at(i)
            if λ > Λ:
                info(
                    f"#{i} halted pool search: objective ({λ:.3f}) > incumbent's length"
                )
                break
            Sʹ = self._topology_from_mip_pool()
            Gʹ = PathFinder(
                G_from_S(Sʹ, A), planar=P, A=A, branched=branched
            ).create_detours()
            Λʹ = Gʹ.size(weight='length')
            if Λʹ < Λ:
                S, G, Λ = Sʹ, Gʹ, Λʹ
                G.graph['pool_entry'] = i, λ
                info(f'#{i} -> incumbent (objective: {λ:.3f}, length: {Λ:.3f})')
            else:
                info(f'#{i} discarded (objective: {λ:.3f}, length: {Λ:.3f})')
        G.graph['pool_count'] = num_solutions
        return S, G
