# CS7641 Machine Learning - Project 2: Randomized Optimization - Francisco Camargo

## Assignment Instructions

The instructions for this assignment can be found within `p2 - Randomized Optimization instructions.pdf`

## Installing environment

Download and install Python 3.11.0 from https://www.python.org/downloads/

To update `pip`, use

`python -m pip install --upgrade pip`

To create an environment via the terminal, use

`python -m venv env`

To activate environment, use

`env/Scripts/activate`

To install libraries, use

`pip install -r requirements.txt`

To deactivate an active environment, use

`deactivate`

## Running the Code

### Part 1 code (3 fitness optima problem solved with 4 randomized optimization algorithms):

Treat the folder `part1_randomized_optimization` as the parent directory. Below the line `if __name__ == '__main__'` inside of `main.py` select the `mode` and the `experiment_name` you would like to run. All the possible choices are written in `main.py`.
To change how a specific experiment runs, go into the corresponding config `.py` file and alter the dictionary. These are found under `./src/hp_search/`, `./src/vsize/`, and `./src/vtime/`.

### Part 2 code (training neural network with various algorithms):

Treat the folder `part2_compare_optimizers` as the parent directory. Below the line `if __name__ == '__main__'` inside of `main.py` select the `experiment_mode`. To select which algorithms will be used during the experiment, change `optimize_algo`, `fitness_algo`, or `learning_algo`.
To change how a specific experiment runs, go into the corresponding config `.py` file and alter the dictionary, these are found within `./src/params/`.

## Hiive MLRose

This repo will use the module `mlrose` from the `hiive` fork https://github.com/hiive/mlrose

PyPi [page](https://pypi.org/project/mlrose-hiive/)

[Documentation](https://mlrose.readthedocs.io/en/stable/) corresponding to the *original* implementation of mlrose, which includes tutorial content

[Here](https://github.com/hiive/mlrose/blob/master/problem_examples.ipynb) is a great notebook with examples of how to use mlrose-hiive.

## Graphs

### Part 1: Explore Randomized Algorithms

We have 3 problems and 4 algorithms.

* Fitness vs iteration
* Fitness vs clock time (per iteration or cumulative)
* Fitness vs FEval (number of evaluations)
* Fitness vs Problem size
* FEval vs Iteration
* Something for hyperparameters

### Part 2: Compare with Gradient Descent

Sounds like we have more freedom here

* Learning curves
* Loss curve; loss vs iteration. Dan: this is preferred... but why
* Wall clock time

## LaTeX template

The LaTeX template I used for this report comes from the Vision Stanford [website](http://vision.stanford.edu/cs598_spring07/report_templates/)
