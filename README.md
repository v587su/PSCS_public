# PSCS (Path-based Semantic Code Search)

This repository contains source code necessary to reproduce the model in the paper:

"PSCS: A Path-based Neural Model for SemanticCode Search"

## Usage

### Requirements

- Python 3.6
- Pytorch 1.0.1
- NumPy

### Dataset

We use the [CodeSearchNet](https://github.com/github/CodeSearchNet) as our dataset.

Download the Java dataset following the instruction of [CodeSearchNet](https://github.com/github/CodeSearchNet), and move it to the directory `./data/`.

The data exists in `./data` is the example data.

### Data processing and Model training

The whole process of our model, including data processing and model training, is in file `pipeline.sh`.

Before running, edit the file `pipeline.sh` following the instructions below.

Run the `pipeline.sh` file:

```
sudo bash pipeline.sh
```

### Evaluation

To evaluate the performance of a trained model:

Run the `test.py` file:

```
python test.py --test_epoch 100
```
