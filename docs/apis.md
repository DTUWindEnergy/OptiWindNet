# Which API?

_OptiWindNet_ offers two ways in, and **learning one of them is enough** — they solve the same problems, with the same routers, and produce the same graphs. This page is the one place that asks you to choose; the rest of the documentation assumes you have.

## 📦 Network/Router API

The {doc}`Network/Router API </high_level_api>` (high level) makes the main functionality of _OptiWindNet_ available through two classes: {py:class}`WindFarmNetwork <optiwindnet.api.WindFarmNetwork>` and {py:class}`Router <optiwindnet.api.Router>`. This approach enables quick experimentation and includes some guardrails for beginners.

- Simple to use, more forgiving on mistakes;
- Gentler learning curve;
- Focused on productivity and ease of interaction;
- Stable: this is the interface covered by the package's compatibility guarantees.

**Recommended for most users.** Start at {doc}`/notebooks/hi00_quickstart`, then work through {doc}`/high_level_api`.

## 🛠️ Advanced API

The {doc}`Advanced API </low_level_api>` (low level) offers fine-grained control of all data structures and functions of _OptiWindNet_.

- Allows picking and choosing exactly what is needed from _OptiWindNet_;
- May perform faster by avoiding unnecessary checks and offering more tuning options;
- The API to use for extending _OptiWindNet_ with custom algorithms, models, objective functions or constraints;
- Developer-facing: these interfaces may evolve independently of the Network/Router API.

Start at {doc}`/notebooks/lo00_quickstart`, then work through {doc}`/low_level_api`.

## Either way

{doc}`/problem` and {doc}`/routers` describe what the tool computes and how, without reference to either API — read them once, whichever you picked. [](/reference/tasks.md#paired-examples) lists the notebooks that exist in both sections, so a task you learned through one API is easy to find in the other.
