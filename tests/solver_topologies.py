# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Expected topologies of the deterministic constructor and MILP cases.

Each entry maps a topology golden key (see
:func:`tests.cases.topology_golden_key`) to the hexadecimal
:func:`~optiwindnet.identity.topology_id` of every topology accepted for it. A
key holds more than one id only where the problem has tied optima that the
producers legitimately disagree on. A change here means a producer selected a
different set of links: review it, do not refresh it to make a test pass.

Regenerate with: python -m tests.update_solver_topologies
"""

__all__ = ('SOLVER_TOPOLOGY_GOLDENS',)

SOLVER_TOPOLOGY_GOLDENS: dict[str, tuple[str, ...]] = {
    'constructor-site-albatros-capacity-3-method-ringed-feeder-route-segmented-bias-margin-none': (
        'de922f401360383e08cf92ad667fe7be',
    ),
    'constructor-site-cazzaro_2022-capacity-5-method-biased-ew-feeder-route-segmented-bias-margin-none': (
        '3a100d44d763de782172ed2ce66236a7',
    ),
    'milp-site-toy-capacity-5-topology-branched-feeder-route-segmented-feeder-limit-unlimited-balanced-false-max-feeders-0': (
        'd6c4791fce89e7b252a11205db708e8c',
    ),
}
