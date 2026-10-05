# CS324 Deep Learning Assignment 1

## Requirements

Use Python 3.11 or newer with:

```bash
python -m pip install numpy scikit-learn matplotlib jupyter nbconvert
```

## Part I

Run the Gaussian/perceptron experiments:

```bash
cd Part_1
python experiments.py
```

## Part II and Part III

Run full-batch gradient descent:

```bash
cd Part_2
python train_mlp_numpy.py --dnn_hidden_units 20 --learning_rate 0.01 --max_steps 1500 --eval_freq 10 --batch_size 0
```

Run stochastic gradient descent:

```bash
python train_mlp_numpy.py --dnn_hidden_units 20 --learning_rate 0.01 --max_steps 1500 --eval_freq 10 --batch_size 1
```

Run the executed experiment notebook:

```bash
jupyter notebook mlp_report_executed.ipynb
```

The notebook contains the training/test accuracy curves, loss curves, and
the comparison of batch sizes required for Part III.
