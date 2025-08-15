# Example Notebooks Overview

OptiWindNet’s example notebooks demonstrate wind farm cable layout optimization in two main categories:

* 📦 **Network/Router API**: high-level, user-friendly interface (recommended for most users)
* 🛠️ **Low-Level API**: advanced, fine-grained control for customization and research


## 📦 Network/Router API Examples

These notebooks demonstrate how to use the *OptiWindNet*'s **high-level API** via the `WindFarmNetwork` and `Router` classes. This approach is:

* Recommended for most users
* Simple to use
* Suitable for integration with tools like TopFarm
* Focused on productivity and ease of interaction

Users only need to provide input data and call a few intuitive methods to perform routing and access results.

The notebooks demonstrating the high-level API of *OptiWindNet* have filenames starting with a letter.

> ✅ Use these if you want fast prototyping or plan to integrate `OptiWindNet` into a larger workflow.

## 🛠️ Low-Level API Examples

These notebooks show how to use *OptiWindNet* by directly importing its internal modules and functions. This approach:

* Exposes more customization and internal logic
* Requires calling several intermediate steps (e.g., preprocessing, initial tree generation, optimization, result processing)
* Is useful for **advanced users**, researchers, or developers who want fine control or want to extend the code

The notebooks demonstrating **low-level API** of *OptiWindNet* are those with filenames starting with a number.

> 🔍 Use these if you're exploring the algorithm, debugging, or building on top of the library internals.
