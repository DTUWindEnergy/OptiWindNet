.. _installation:

.. Installation Guide
.. ==================

Requirements
------------
A recent Python version (3.10+) is required to run *OptiWindNet*, and we recommend to install it in its own virtual environment. This can be achieved by installing **either**:

* `Python <https://www.python.org/downloads/>`_ and using the built-ins ``venv`` virtual environment creator and ``pip`` package manager;
* or `Miniforge <https://conda-forge.org/download/>`_ (`Anaconda or Miniconda <https://www.anaconda.com/download/success>`_ also work) and using ``conda`` to create and populate the virtual environment.

In the near future OptiWindNet will be turned into a Python package installable with the usual package managers, but currently it must be installed from the project tree obtained via ``git``. This software can be obtained from `Git <https://git-scm.com/downloads>`_ for a standalone version or from `Git for Windows <https://gitforwindows.org/>`_ to get bundle of git and other useful tools for the Windows platform (recommended).

Installation
------------
The following commands must be run from the system's command line interface (e.g. *git-bash*, *cmd*, *powershell*), first making sure that `git` and `python` or `conda` are available (see section above)::

    git clone https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet.git

If using ``venv``/``pip``
^^^^^^^^^^^^^^^^^^^^^^^^^

Run::

    python -m venv optiwindnet_env

* cmd: ``optiwindnet_env\Scripts\activate.bat``
* bash: ``source optiwindnet_env/Scripts/activate``
* powershell: ``optiwindnet_env\Scripts\Activate.ps1``

Then::

    pip install -r OptiWindNet/requirements.txt
    pip install --editable OptiWindNet/


If using ``conda``
^^^^^^^^^^^^^^^^^^

Run::

    conda env create -f OptiWindNet/environment.yml
    activate optiwindnet_env
    pip install --editable OptiWindNet/

Updating
--------

Run::

    cd OptiWindNet
    git pull

If ``pip`` was given the ``--editable`` option when installing, the new version will be immediately available within the Python environment ``optiwindnet_env``.
