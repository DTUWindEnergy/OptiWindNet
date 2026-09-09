# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Regenerate the expected topologies of the deterministic producer cases.

Run from the repository root with::

    python -m tests.update_solver_topologies

Every exact-golden constructor case is built and the required MILP case is
solved; each result is reduced to the hexadecimal ``topology_id`` recorded in
``tests/solver_topologies.py``. A key whose golden accepts more than one id
keeps the extra ones, since only the produced id is regenerated -- review the
diff instead of accepting it.
"""

import argparse
from pathlib import Path

from .cases import (
    CONSTRUCTOR_CASES,
    MILP_FORMULATION_CASES,
    expected_topology,
    golden_keys,
    topology_golden_key,
)
from .helpers import solve_milp_case
from .producers import constructor_topology
from .solver_topologies import SOLVER_TOPOLOGY_GOLDENS
from .topology_assertions import assert_topology

SOLVER_TOPOLOGIES_FILE = Path(__file__).with_name('solver_topologies.py')

_TEMPLATE = '''"""Expected topologies of the deterministic constructor and MILP cases.

Each entry maps a topology golden key (see
:func:`tests.cases.topology_golden_key`) to the hexadecimal
:func:`~optiwindnet.identity.topology_id` of every topology accepted for it. A
key holds more than one id only where the problem has tied optima that the
producers legitimately disagree on. A change here means a producer selected a
different set of links: review it, do not refresh it to make a test pass.

Regenerate with: python -m tests.update_solver_topologies
"""

__all__ = ('SOLVER_TOPOLOGY_GOLDENS',)

SOLVER_TOPOLOGY_GOLDENS: dict[str, tuple[str, ...]] = {{
{entries}}}
'''


def generate() -> dict[str, tuple[str, ...]]:
    """Produce every exact-golden case and reduce it to its accepted ids."""
    goldens = {}
    for case in CONSTRUCTOR_CASES:
        if not case.exact_golden:
            continue
        S = constructor_topology(case)
        assert_topology(S, expected_topology(case), case.capacity)
        goldens[topology_golden_key(case)] = S.graph['_topology_id']

    # The primary required solver owns the shared adapter problem golden.
    case = next(case for case in MILP_FORMULATION_CASES if case.exact_golden)
    info, S = solve_milp_case(case)
    assert_topology(S, case.model_options['topology'], case.capacity)
    goldens[topology_golden_key(case)] = info.topology_id

    if goldens.keys() != golden_keys():
        missing = sorted(golden_keys() - goldens.keys())
        stale = sorted(goldens.keys() - golden_keys())
        raise ValueError(f'golden keys differ: missing={missing}, stale={stale}')
    # the produced id leads; any other id already accepted for the key follows
    return {
        key: tuple(
            dict.fromkeys((produced.hex(), *SOLVER_TOPOLOGY_GOLDENS.get(key, ())))
        )
        for key, produced in goldens.items()
    }


def render(goldens: dict[str, tuple[str, ...]]) -> str:
    """Return the source of the golden module, one accepted id per line."""
    entries = ''.join(
        f'    {key!r}: (\n'
        + ''.join(f'        {item!r},\n' for item in ids)
        + '    ),\n'
        for key, ids in sorted(goldens.items())
    )
    return _TEMPLATE.format(entries=entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=SOLVER_TOPOLOGIES_FILE)
    args = parser.parse_args()
    goldens = generate()
    args.output.write_text(render(goldens))
    print(f'Saved {len(goldens)} topology goldens to {args.output}')


if __name__ == '__main__':
    main()
