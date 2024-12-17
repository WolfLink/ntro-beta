# Numerical T Gate Reduction BQSKit Pass

This software implementation the awesome Numerical T Gate Reduction(link) work by
Marc Davis as an optimization pass in BQSKit.

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
