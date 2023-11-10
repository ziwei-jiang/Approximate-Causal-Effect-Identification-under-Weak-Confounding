Approximate-Causal-Effect-Identification-under-Weak-Confounding
============================
This repository contains the code for our paper [Approximate Causal Effect Identification under Weak Confounding](https://proceedings.mlr.press/v202/jiang23h) ([arXiv](https://arxiv.org/pdf/2306.13242.pdf)).

Requirement
----------------------------
* Argparse
* Numpy
* Scipy
* CXVpy
* Matplotlib
* tqdm
* pickle

Usage
----------------------------
run the script partial_identification_with_entropy.py with value P(y|x), P(x), and H(U).
 
```
$ partial_identification_with_entropy.py [-h] --y_x Y_X --x X [--entr ENTR]

options:
  -h, --help   show this help message and exit
  --y_x Y_X    The conditional probability P(y|x) that corresponding to p(y|do(x))
  --x X        The marginal probability P(x) that corresponding to P(y|do(x))
  --entr ENTR  The upper bound of confounder entropy

```