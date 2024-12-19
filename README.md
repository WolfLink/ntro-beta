# Numerical T Gate Reduction BQSKit Pass

This is a Python implementation of the Rz and T gate minimization technique described in [Numerical synthesis of arbitrary multi-qubit unitaries with low T-count](https://mit.primo.exlibrisgroup.com/permalink/01MIT_INST/ejdckj/alma9935460727606761).  It uses numerical optimization to tune the angles of Rz gates in a quantum circuit, with the goal of tuning these Rz gates such that they can be replaced by Clifford gates or Clifford gates with one T gate when possible.  In a typical use case, any Rz gates not rounded this way will be approximated using a technique such as the [Selinger-Ross `gridsynth` algorithm](https://arxiv.org/abs/1403.2975).

This package is designed as a pass for BQSKit, adding Clifford+T optimizing tools to BQSKit's quantum synthesis toolkit.  You will find instructions below on how to incorporate the `gridsynth` tool to perform the last step of approximating any remaining Rz gates.

## Prerequisites
- BQSKit (https://github.com/BQSKit/bqskit)
- gridsynth (optional) (https://www.mathstat.dal.ca/~selinger/newsynth/)

## Installation

This is available for Python 3.8+ on Linux, macOS, and Windows.

```sh
git clone https://github.com/WolfLink/ntro
pip install ./ntro
```

## Basic Usage

NTRO provides tools to be used in a `BQSKit` workflow to convert circuits to the Clifford+T gate set with a minimized T count.

- `NumericalTReductionPass`: This is a bqskit pass that will tweak the parameters of Rz gates, attempting to round as many gates as possible to Clifford or T gates.
- `GridsynthPass`: This is a pass that uses [`gridsynth`](https://www.mathstat.dal.ca/~selinger/newsynth/) to convert any remaining Rz gates to Clifford+T.  You must acquire a `gridsynth` binary, which can be downloaded from the gridsynth website.  We also provide a simple script to build gridsynth from source within a Docker container in `ntro/ntro/newsynth`.

For an example of how to use these passes in a BQSKit workflow, see `ntro/examples/qft_synthesis.py`
