"""Deploy reviewed provisional MILP references without rerunning any solver.

Run from the repository root with::

    python -m tests.update_milp_references

The output is a dictionary keyed by the solver-independent problem key. Each
value keeps the proven bound, objective, and topology together as one atomic
reference record.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from .milp_reference_testing import (
    MILP_REFERENCES_FILE,
    PROVISIONAL_REFERENCES_FILE,
    MILPReference,
    load_provisional_milp_references,
)


def deploy_references(
    source: Path = PROVISIONAL_REFERENCES_FILE,
    output: Path = MILP_REFERENCES_FILE,
) -> dict[str, MILPReference]:
    """Validate provisional results and pickle them without solving again."""
    references = load_provisional_milp_references(source)
    with output.open('wb') as file:
        pickle.dump(references, file, protocol=pickle.HIGHEST_PROTOCOL)
    return references


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=PROVISIONAL_REFERENCES_FILE)
    parser.add_argument('--output', type=Path, default=MILP_REFERENCES_FILE)
    args = parser.parse_args()
    references = deploy_references(args.source, args.output)
    print(f'Saved {len(references)} MILP references to {args.output}')


if __name__ == '__main__':
    main()
