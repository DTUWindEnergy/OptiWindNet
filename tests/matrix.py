"""Declarative end-to-end test matrix.

Single source of truth for the (site x router) combinations the end-to-end suite
exercises. Specs are built from explicit option axes via the ``ew_spec`` /
``hgs_spec`` / ``milp_spec`` factories, so widening coverage -- a new topology,
solver, feeder-limit or heuristic method -- is a one-line change instead of a
hand-written dict per combination.

Every spec carries a ``determinism`` tag that selects the assertion strategy in
``test_end_to_end.py``:

  ``'snapshot'``
      Deterministic backend (EWRouter -- a pure greedy heuristic, no RNG). The
      exact ``terse_links`` are pinned in ``solutions.pkl`` and compared.

  ``'property'``
      Backend whose exact output is not reproducible across solver versions,
      seeds or hardware (MILP optima are non-unique / gap-dependent; HGS is a
      metaheuristic). The test asserts the solution's *properties* -- validity,
      topology shape, capacity, feeder/balance constraints -- and, when a
      reference optimum was recorded, that the objective length is within the
      solver's gap tolerance of it. No exact snapshot is stored.

The matrix is intentionally a *curated* cross-section, not a full Cartesian
product: every axis value appears, with dense cross-coverage on the cheap
``example_location`` site and representative breadth on larger, multi-substation
sites. Site handles below must exist in ``tests/sites.py``.
"""

# --------------------------------------------------------------------------- #
# Axes
# --------------------------------------------------------------------------- #
EW_METHODS = ('biased_EW', 'radial_EW', 'rootlust', 'esau_williams', 'ringed')
FEEDER_ROUTES = ('segmented', 'straight')
TOPOLOGIES = ('branched', 'radial', 'ringed')
# feeder-limit values that need no max_feeders (usable site-agnostically)
FEEDER_LIMITS = ('unlimited', 'minimum', 'min_plus1')
# every MILP backend optiwindnet exposes; absent ones are skipped by the test.
MILP_SOLVERS = (
    'ortools.cp_sat',
    'ortools.gscip',
    'ortools.highs',
    'highs',
    'scip',
    'gurobi',
    'cplex',
    'cbc',
    'fscip',
)
# a single reliable solver used to sweep the model-option axes densely
SWEEP_SOLVER = 'ortools.cp_sat'


# --------------------------------------------------------------------------- #
# Spec factories
# --------------------------------------------------------------------------- #
def ew_spec(cables, *, method='biased_EW', feeder_route='segmented', bias_margin=None):
    params = {}
    if method != 'biased_EW':
        params['method'] = method
    if feeder_route != 'segmented':
        params['feeder_route'] = feeder_route
    if bias_margin is not None:
        params['bias_margin'] = bias_margin
    # EWRouter is deterministic, but the terse_links snapshot format is
    # forest-only (one parent per node) and cannot encode a ring's two-feeder
    # node. Ringed EW is therefore validated by properties, like MILP/HGS.
    determinism = 'property' if method == 'ringed' else 'snapshot'
    return {
        'class': 'EWRouter',
        'params': params,
        'cables': int(cables),
        'determinism': determinism,
    }


def hgs_spec(
    cables,
    *,
    time_limit=0.5,
    seed=0,
    ringed=False,
    balanced=False,
    feeder_limit=None,
    feeder_exact=False,
):
    params = {'time_limit': time_limit, 'seed': seed}
    if ringed:
        params['ringed'] = True
    if balanced:
        params['balanced'] = True
    if feeder_limit is not None:
        params['feeder_limit'] = feeder_limit
    if feeder_exact:
        params['feeder_exact'] = True
    return {
        'class': 'HGSRouter',
        'params': params,
        'cables': int(cables),
        'determinism': 'property',
    }


def milp_spec(
    solver,
    cables,
    *,
    topology='branched',
    feeder_route='segmented',
    feeder_limit='unlimited',
    balanced=False,
    max_feeders=0,
    time_limit=5,
    mip_gap=1e-3,
):
    model_options = {}
    if topology != 'branched':
        model_options['topology'] = topology
    if feeder_route != 'segmented':
        model_options['feeder_route'] = feeder_route
    if feeder_limit != 'unlimited':
        model_options['feeder_limit'] = feeder_limit
    if balanced:
        model_options['balanced'] = True
    if max_feeders:
        model_options['max_feeders'] = max_feeders
    params = {'solver_name': solver, 'time_limit': time_limit, 'mip_gap': mip_gap}
    if model_options:
        params['model_options'] = model_options
    return {
        'class': 'MILPRouter',
        'params': params,
        'cables': int(cables),
        'determinism': 'property',
    }


# --------------------------------------------------------------------------- #
# Stable keys
# --------------------------------------------------------------------------- #
def spec_key(spec):
    """A deterministic, human-readable key for a router spec (no site prefix)."""
    cls = spec['class']
    p = spec['params']
    parts = []
    if cls == 'EWRouter':
        parts += ['ew', p.get('method', 'biased_EW'), p.get('feeder_route', 'segmented')]
        if 'bias_margin' in p:
            parts.append('bm' + str(p['bias_margin']).replace('.', 'p'))
    elif cls == 'HGSRouter':
        parts.append('hgs')
        parts.append('ringed' if p.get('ringed') else 'radial')
        if p.get('balanced'):
            parts.append('balanced')
        if p.get('feeder_exact'):
            parts.append('fexact')
        if p.get('feeder_limit') is not None:
            parts.append('fl' + str(p['feeder_limit']))
    elif cls == 'MILPRouter':
        mo = p.get('model_options', {})
        parts += [
            'milp',
            p['solver_name'].replace('.', '_'),
            mo.get('topology', 'branched'),
            mo.get('feeder_route', 'segmented'),
            mo.get('feeder_limit', 'unlimited'),
        ]
        if mo.get('balanced'):
            parts.append('bal')
        if mo.get('max_feeders'):
            parts.append('mf' + str(mo['max_feeders']))
    else:
        raise ValueError(f'unknown router class {cls!r}')
    parts.append(f'cap{spec["cables"]}')
    return '_'.join(parts)


def case_key(site, spec):
    return f'{site}__{spec_key(spec)}'


def problem_key(site, spec):
    """A solver-independent key for a MILP *problem* (site, cables, options).

    Every MILP solver run on the same problem targets the same optimum, so a
    single reference objective length is shared across them under this key.
    """
    mo = spec['params'].get('model_options', {})
    opt_part = '_'.join(f'{k}={mo[k]}' for k in sorted(mo)) or 'default'
    return f'{site}__cap{spec["cables"]}__{opt_part}'


# --------------------------------------------------------------------------- #
# The curated matrix
# --------------------------------------------------------------------------- #
def _ew_cases():
    """EWRouter: every method x feeder_route on the small site, plus breadth."""
    cases = []
    small = 'example_location'
    # every method, both feeder routes, at a mid capacity
    for method in EW_METHODS:
        for feeder_route in FEEDER_ROUTES:
            cases.append((small, ew_spec(3, method=method, feeder_route=feeder_route)))
    # capacity spread on the default and ringed methods
    for cables in (1, 10):
        cases.append((small, ew_spec(cables)))
        cases.append((small, ew_spec(cables, method='ringed')))
    # a bias_margin variant (ringed uses it distinctly)
    cases.append((small, ew_spec(3, method='ringed', bias_margin=0.1)))
    # breadth on larger, multi-substation sites (default + ringed)
    for site in ('london', 'hornsea', 'taylor_2023', 'yi_2019'):
        cases.append((site, ew_spec(10)))
        cases.append((site, ew_spec(10, method='ringed')))
        cases.append((site, ew_spec(10, feeder_route='straight')))
    return cases


def _hgs_cases():
    """HGSRouter: radial / ringed / balanced / feeder-limited options."""
    cases = []
    small = 'example_location'
    cases += [
        (small, hgs_spec(3)),
        (small, hgs_spec(3, ringed=True)),
        (small, hgs_spec(4, balanced=True)),
        (small, hgs_spec(4, feeder_limit=0)),
        (small, hgs_spec(3, ringed=True, balanced=True)),
    ]
    # breadth: radial and ringed on a larger multi-substation site
    cases += [
        ('london', hgs_spec(10, time_limit=1.0)),
        ('london', hgs_spec(10, ringed=True, time_limit=1.0)),
    ]
    return cases


def _milp_cases():
    """MILPRouter: dense model-option sweep + per-solver breadth."""
    cases = []
    small = 'example_location'

    # --- dense model-option sweep on one reliable solver (small site) --------
    # topology x feeder_route
    for topology in TOPOLOGIES:
        for feeder_route in FEEDER_ROUTES:
            cases.append(
                (small, milp_spec(SWEEP_SOLVER, 3, topology=topology,
                                  feeder_route=feeder_route))
            )
    # feeder_limit axis (branched)
    for feeder_limit in FEEDER_LIMITS:
        cases.append(
            (small, milp_spec(SWEEP_SOLVER, 3, feeder_limit=feeder_limit))
        )
    # feeder_limit values that need an explicit max_feeders (example: T=12,
    # cap=3 => min_feeders=4)
    cases.append(
        (small, milp_spec(SWEEP_SOLVER, 3, feeder_limit='exactly', max_feeders=5))
    )
    cases.append(
        (small, milp_spec(SWEEP_SOLVER, 3, feeder_limit='specified', max_feeders=6))
    )
    # balanced requires a pinned feeder count (minimum)
    cases.append(
        (small, milp_spec(SWEEP_SOLVER, 3, feeder_limit='minimum', balanced=True))
    )
    cases.append(
        (
            small,
            milp_spec(SWEEP_SOLVER, 3, topology='ringed', feeder_limit='minimum',
                      balanced=True),
        )
    )

    # --- per-solver breadth: every backend sees branched + radial + ringed,
    #     plus a straight-feeder + pinned-feeder-count case. The straight route
    #     and feeder-limit constraints are solver-specific code (each backend has
    #     its own get_solution() STRAIGHT path and constraint emission), so the
    #     segmented+unlimited cases alone leave those branches uncovered.
    for solver in MILP_SOLVERS:
        cases.append((small, milp_spec(solver, 5, topology='branched')))
        cases.append((small, milp_spec(solver, 5, topology='radial')))
        cases.append((small, milp_spec(solver, 5, topology='ringed')))
        cases.append(
            (small, milp_spec(solver, 5, topology='radial',
                              feeder_route='straight', feeder_limit='minimum'))
        )

    # --- a couple of larger / multi-substation MILP cases --------------------
    cases.append(('borkum2', milp_spec(SWEEP_SOLVER, 5, topology='ringed',
                                       time_limit=10)))
    cases.append(('taylor_2023', milp_spec(SWEEP_SOLVER, 8, time_limit=10)))
    return cases


def build_matrix():
    """Return the full curated matrix as a list of ``(site, spec)`` pairs.

    Semantically identical specs reached from different axis loops (e.g. the
    ``(branched, segmented)`` topology-sweep case and the ``unlimited``
    feeder-limit case) collapse to one entry, keeping the first occurrence.
    """
    seen = set()
    unique = []
    for site, spec in _ew_cases() + _hgs_cases() + _milp_cases():
        key = case_key(site, spec)
        if key not in seen:
            seen.add(key)
            unique.append((site, spec))
    return unique


def matrix_by_key():
    """Return the matrix as an ordered ``{case_key: (site, spec)}`` dict."""
    out = {}
    for site, spec in build_matrix():
        out[case_key(site, spec)] = (site, spec)
    return out
