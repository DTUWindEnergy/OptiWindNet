# syntax=docker/dockerfile:1
# Build context: the OptiWindNet repository root.
# To reuse a published environment, replace `environment` below with its
# public registry reference (preferably pinned by digest).
ARG ENVIRONMENT_IMAGE=environment
ARG JUPYTER_IMAGE=quay.io/jupyter/base-notebook:python-3.13

# Compile dependencies here when wheels are unavailable. Compilers and
# the wheelhouse are excluded from the final image.
FROM ${JUPYTER_IMAGE} AS wheels
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*

# Keep dependency installation independent of notebook/source edits.
COPY pyproject.toml /tmp/optiwindnet-pyproject.toml
RUN python - <<'PY'
import pathlib
import tomllib

p = tomllib.loads(
    pathlib.Path("/tmp/optiwindnet-pyproject.toml").read_text()
)
requirements = (
    p["project"]["dependencies"]
    + p["project"]["optional-dependencies"]["solvers"]
    + p["build-system"]["requires"]
    + ["ipywidgets", "jupyterlab-execute-time", "wheel", "prettyTables"]
)
pathlib.Path("/tmp/requirements.txt").write_text(
    "\n".join(requirements) + "\n"
)
PY

RUN python -m pip wheel --prefer-binary --no-cache-dir \
      --wheel-dir=/tmp/wheels -r /tmp/requirements.txt

FROM ${JUPYTER_IMAGE} AS environment
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends git libgomp1 libstdc++6 \
 && rm -rf /var/lib/apt/lists/*

# Mount build-stage wheels without adding them to an image layer.
RUN --mount=from=wheels,source=/tmp,target=/tmp/wheel-build,ro \
    python -m pip install --no-cache-dir --no-index \
      --find-links=/tmp/wheel-build/wheels \
      -r /tmp/wheel-build/requirements.txt \
 && python -m pip check \
 && fix-permissions /opt/conda

USER jovyan
WORKDIR /home/jovyan
# Unset Jupyter Stack startup wrappers; Binder supplies its own command.
ENTRYPOINT []
CMD ["jupyter", "lab", "--ip=0.0.0.0"]

FROM ${ENVIRONMENT_IMAGE} AS binder
USER jovyan
WORKDIR /home/jovyan
COPY --chown=1000:100 . /home/jovyan
# Dependencies and the build backend are already in the environment.
# Keep .git in the context so setuptools-scm can determine the version.
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation -e . \
 && python -m pip check \
 && python -c 'import optiwindnet, prettyTables, pyscipopt, cbcbox, highspy'

ENTRYPOINT []
CMD ["jupyter", "lab", "--ip=0.0.0.0"]
