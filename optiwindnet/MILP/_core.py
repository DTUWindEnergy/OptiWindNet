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
                + ', '.join(f'"{m}"' for m in v.__members__.values() if m != 'default')
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
    solver_options: dict[str, Any]
    solution_info: SolutionInfo

    @abc.abstractmethod
    def set_problem(
        self,
        P: nx.PlanarEmbedding,
        A: nx.Graph,
        capacity: int,
        model_options: ModelOptions,
    ):
        """Define the problem geometry, available edges and tree properties

        Args:
          P: planar embedding of the location
          A: available edges for the location
          capacity: maximum number of terminals in a subtree
          model_options: tree properties - see ModelOptions.help()
        """
        pass

    @abc.abstractmethod
    def solve(
        self,
        time_limit: int,
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
    def get_solution(self, A: nx.Graph | None) -> tuple[nx.Graph, nx.Graph]:
        """Output solution topology A and routeset G.

        Args:
          A: optionally replace the A given via set_problem() (if normalized A)

        Returns:
          Topology graph S and routeset G.
        """
        pass

    def _make_graph_attributes(self) -> dict[str, Any]:
        metadata, solution_info = self.metadata, self.solution_info
        attr = dict(
            **asdict(solution_info),
            method_options=dict(
                solver_name=self.name,
                fun_fingerprint=metadata.fun_fingerprint,
                **self.solver_options,
                **metadata.model_options,
            ),
        )
        if metadata.warmed_by:
            attr['warmstart'] = metadata.warmed_by
        return attr


class PoolHandler(abc.ABC):
    name: str
    num_solutions: int
    model_options: ModelOptions

    @abc.abstractmethod
    def objective_at(self, index: int) -> float:
        "Get objective value from solution pool at position `index`"
        pass

    @abc.abstractmethod
    def topology_from_mip_pool(self) -> nx.Graph:
        "Build topology from the pool solution at the last requested position"
        pass


def investigate_pool(
    P: nx.PlanarEmbedding, A: nx.Graph, pool: PoolHandler
) -> tuple[nx.Graph, nx.Graph]:
    """Go through the solver's solutions checking which has the shortest length
    after applying the detours with PathFinder."""
    Λ = float('inf')
    branched = pool.model_options['topology'] is Topology.BRANCHED
    num_solutions = pool.num_solutions
    info(f'Solution pool has {num_solutions} solutions.')
    for i in range(num_solutions):
        λ = pool.objective_at(i)
        if λ > Λ:
            info(f"#{i} halted pool search: objective ({λ:.3f}) > incumbent's length")
            break
        Sʹ = pool.topology_from_mip_pool()
        Sʹ.graph['creator'] += '.' + pool.name
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
