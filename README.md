# causal-inference-ex

Examples of Causal Inference in Python

There is a causalinference library in Python but it has not been updated in
many years.

## Terms

CATE - conditional average treatment effect

* Observed factual outcome
* Unobserved counterfactual outcome

### Propensity score matching

In the statistical analysis of observational data, propensity score matching (PSM) is a statistical matching technique that attempts to estimate the effect of a treatment by accounting for the covariates that predict receiving the treatment. PSM attempts to reduce the bias due to confounding variables that could be found in an estimate of the treatment effect obtained from simply comparing outcomes among units that received the treatment versus those that did not.

## Setup

### Build the Causal inference for the Brave and True book

1. Clone the book repository in the project root

```
git clone https://github.com/matheusfacure/python-causality-handbook
```

2. Build the book

### Chapters

#### 04 - Graphical causal model

```
A -> B -> C  # A independent of C | B
```

Fork structure

```
C -> A and C -> B  # A independent of B | C
```

Collider

```
A -> C and B -> C  # A independent of B and A not independent of B | C
```

Three structures that can lead to bias

1. Confounding bias: treatment and outcome have a common cause that we don't control

2. Selection bias: conditioning on a common effect

3. Selection bias: excessive controlling of mediator variables

#### 08 - Instrumental variables

The `linearmodels` library supports Instrument Variables (IV) models.

## To create the slides

1. Start Anki

2. Run the command to create flashcards

```
python d:\ws\anki-md-deck\Obsidian_to_Anki-3.4.2\obsidian_to_anki.py .\anki-slides.md
```

## Links

### MIT OpenCourseWare

[Causal Inference, Part 1][1000]

[1000]: https://www.youtube.com/watch?v=gRkUhg9Wb-I

[Causal Inference, Part 2][1010]

[1010]: https://www.youtube.com/watch?v=g5v-NvNoJQQ

### Causal inference for the Brave and True

Book:

https://matheusfacure.github.io/python-causality-handbook/landing-page.html

Github repo:

https://github.com/matheusfacure/python-causality-handbook

### 6 free machine learning books

- Deep Learning - https://www.deeplearningbook.org/
- Dive into Deep Learning - d2l.ai
- Machine Learning Engineering - http://www.mlebook.com/wiki/doku.php
- Python Data Science Handbook - https://jakevdp.github.io/PythonDataScienceHandbook/
- Probabilistic Machine Learning - https://probml.github.io/pml-book/book1.html
- Machine Learning Yearning - https://www.deeplearning.ai/resources/#ebooks
