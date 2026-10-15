# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import ast
import importlib.util
import sys
from pathlib import Path

import darkdetect
import pytest

DOCS_DIR = Path(__file__).resolve().parent.parent / 'docs'
if str(DOCS_DIR) not in sys.path:
    sys.path.insert(0, str(DOCS_DIR))
spec = importlib.util.spec_from_file_location(
    'run_notebooks', DOCS_DIR / 'run_notebooks.py'
)
assert spec is not None and spec.loader is not None
run_notebooks = importlib.util.module_from_spec(spec)
sys.modules['run_notebooks'] = run_notebooks
spec.loader.exec_module(run_notebooks)


def test_theme_argument_parsing():
    parser = run_notebooks.argument_parser()
    args = parser.parse_args(['--kernel', 'python3'])
    assert args.theme is None

    args = parser.parse_args(['--kernel', 'python3', '--theme', 'dark'])
    assert args.theme == 'dark'

    args = parser.parse_args(['--kernel', 'python3', '--theme', 'Light'])
    assert args.theme == 'light'

    args = parser.parse_args(['--kernel', 'python3', '--darkdetect-theme', 'DARK'])
    assert args.theme == 'dark'

    with pytest.raises(SystemExit):
        parser.parse_args(['--kernel', 'python3', '--theme', 'sepia'])


@pytest.mark.parametrize(
    ('theme_arg', 'expected_is_dark', 'expected_theme'),
    [
        ('dark', True, 'Dark'),
        ('light', False, 'Light'),
    ],
)
def test_darkdetect_exec_line_execution(
    monkeypatch, theme_arg, expected_is_dark, expected_theme
):
    # the line rebinds darkdetect's functions: monkeypatch restores them
    for name in ('theme', 'isDark', 'isLight'):
        monkeypatch.setattr(darkdetect, name, getattr(darkdetect, name))
    namespace: dict = {'__name__': '__main__'}
    exec(run_notebooks.darkdetect_exec_line(theme_arg), namespace)  # noqa: S102

    assert darkdetect.isDark() is expected_is_dark
    assert darkdetect.isLight() is (not expected_is_dark)
    assert darkdetect.theme() == expected_theme
    # the startup line leaves no name in the kernel's namespace
    assert '_darkdetect' not in namespace


def test_kernel_arguments():
    parser = run_notebooks.argument_parser()

    args = parser.parse_args(['--kernel', 'python3'])
    assert run_notebooks.kernel_arguments(args) == []

    args = parser.parse_args(['--kernel', 'python3', '--theme', 'dark'])
    (argument,) = run_notebooks.kernel_arguments(args)
    option, value = argument.split('=', 1)
    assert option == '--IPKernelApp.exec_lines'
    assert ast.literal_eval(value) == [run_notebooks.darkdetect_exec_line('dark')]
