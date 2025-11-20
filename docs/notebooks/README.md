# Example Notebooks Overview

This repository includes a set of example notebooks to demonstrate how to use `OptiWindNet` for wind farm electrical network optimization. There are **two categories** of notebooks:

## 📦 Network/Router API (Recommended for Most Users)

These notebooks demonstrate how to use the *OptiWindNet*'s **high-level API** via the `WindFarmNetwork` and `Router` classes. This approach is:

* **Simple** to use
* **Suitable for integration** with tools like **TopFarm**
* Focused on productivity and ease of interaction

Users only need to provide input data and call a few intuitive methods to perform routing and access results.

The notebooks demonstrating the **Network/Router API** of *OptiWindNet* have filenames starting with a letter.

> ✅ Use these if you want fast prototyping or plan to integrate `OptiWindNet` into a larger workflow.

## 🛠️ Advanced API Examples

These notebooks show how to use *OptiWindNet* by directly importing its internal modules and functions. This approach:

* Exposes more customization and internal logic
* Requires calling several intermediate steps (e.g., preprocessing, initial tree generation, optimization, result processing)
* Is useful for **advanced users**, researchers, or developers who want fine control or want to extend the code

The notebooks demonstrating **Advanced API** of *OptiWindNet* are those with filenames starting with a number.

> 🔍 Use these if you're exploring the algorithm, debugging, or building on top of the library internals.
