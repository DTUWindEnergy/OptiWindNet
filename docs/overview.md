# APIs Overview

OptiWindNet’s documentation is based on two series of example Jupyter notebooks, each with a specific focus:

* 📦 **Network/Router API**: high-level, user-friendly interface (recommended for most users)
* 🛠️ **Advanced API**: low-level, fine-grained control for performance, customization and research

The notebooks from the two sections show similar tasks performed with each API. Learning only **one of them is enough** to start using *OptiWindNet*.

## 📦 Network/Router API

The [**Network/Router API**](high_level_api) (high level) makes the main functionality of *OptiWindNet* available through two classes: `WindFarmNetwork` and `Router`. This approach enables quick experimentation and includes some guardrails for beginners.

* Simple to use, more forgiving on mistakes;
* Gentler learning curve;
* Focused on productivity and ease of interaction.

## 🛠️ Advanced API

The [**Advanced API**](low_level_api) (low level) offers fine-grained control of all data structures and functions of *OptiWindNet*. 

* Allows picking and choosing exactly what is needed from *OptiWindNet*;
* May perform faster by avoiding unecessary checks and offering more tuning options.
* The API to use for extending *OptiWindNet* with custom algorithms, models, objective functions or constraints.
