# Self-Pruning-Neural-Network


## Overview

This project implements a **Self-Pruning Neural Network** that learns to remove its own unnecessary weights during training. Instead of pruning a model **after training**, the network dynamically learns which connections are important by attaching a **learnable gate parameter to each weight**.

During training, a sparsity penalty encourages many gates to move toward **zero**, effectively removing the corresponding weights. The result is a **smaller and more efficient neural network** that maintains competitive accuracy while reducing computational cost.

This project was implemented as part of an **AI Engineering case study**, demonstrating model design, custom layer implementation, and experimental analysis.

---

## Key Idea

Each weight ( w_{ij} ) in a neural network is associated with a learnable gate value:

[
g_{ij} = \sigma(s_{ij})
]

where:

* ( s_{ij} ) = gate score (trainable parameter)
* ( \sigma ) = sigmoid function

The effective weight used during forward propagation becomes:

[
\tilde{w}*{ij} = w*{ij} \times g_{ij}
]

If ( g_{ij} \approx 0 ), the connection is effectively **pruned**.

---

## Loss Function

The training objective combines classification accuracy with sparsity regularization:

[
\text{Total Loss} = \text{CrossEntropyLoss} + \lambda \times \text{SparsityLoss}
]

Where:

* **CrossEntropyLoss** → classification objective
* **SparsityLoss** → L1 penalty on gate values
* **λ (lambda)** → controls the trade-off between accuracy and sparsity

A higher λ encourages stronger pruning.

---

## Model Architecture

Input images from CIFAR-10 are flattened and passed through a feed-forward network with prunable layers:

```
Input (32 × 32 × 3)
        ↓
PrunableLinear (3072 → 1024)
BatchNorm + ReLU + Dropout
        ↓
PrunableLinear (1024 → 512)
BatchNorm + ReLU + Dropout
        ↓
PrunableLinear (512 → 10)
        ↓
Softmax (via CrossEntropyLoss)
```

Each **PrunableLinear layer** replaces the standard linear layer and includes learnable gates.

---

## Training Strategy

Several techniques are used to stabilize training and improve pruning performance:

* **Negative initialization of gate scores**
* **Temperature annealing** for sharper gate decisions
* **Separate learning rates** for weights and gate parameters
* **Cosine learning rate scheduling**
* **Gradient clipping**
* **Batch normalization and dropout**

These techniques ensure the model learns both **accurate classification and effective sparsity**.

---

## Experimental Setup

Dataset: **CIFAR-10**

* 50,000 training images
* 10,000 test images
* 10 object classes

Data augmentation used:

* Random horizontal flip
* Random cropping
* Normalization using CIFAR-10 statistics

Experiments were run with multiple values of λ to analyze the **accuracy-sparsity trade-off**.

Example λ values tested:

```
λ = 0.5
λ = 1.0
λ = 2.0
```

---

## Evaluation Metrics

The following metrics are reported:

* **Test Accuracy**
* **Sparsity Level (%)**
* **Parameter Compression**
* **FLOPs Reduction**

A weight is considered pruned if:

```
gate < 0.01
```

---

## Example Results

| Lambda | Test Accuracy | Sparsity |
| ------ | ------------- | -------- |
| 0.5    | ~57%          | ~0%      |
| 1.0    | ~57%          | ~3%      |
| 2.0    | ~56%          | ~16%     |

These results demonstrate the expected **trade-off between accuracy and model sparsity**.

---

## Visualizations

The project generates several plots to analyze pruning behavior:

1. **Gate Value Distribution**
   Shows how many gates move toward zero.

2. **Sparsity vs Lambda**
   Demonstrates the effect of the sparsity coefficient.

3. **Training Loss Curves**
   Tracks classification and sparsity losses during training.

4. **Validation Accuracy Curve**

5. **Sparsity Growth During Training**

These visualizations help understand **how and when pruning occurs**.

---

## Project Structure

```
self-pruning-network
│
├── main.py
├── models.py
├── training.py
├── utils.py
│
├── plots
│   ├── gate_distribution.png
│   ├── sparsity_vs_lambda.png
│   ├── training_curves.png
│   └── sparsity_growth.png
│
└── README.md
```

---

## How to Run

Install dependencies:

```
pip install torch torchvision matplotlib numpy tqdm
```

Run training:

```
python main.py
```

The script will:

1. Train the model for different λ values
2. Save the best model checkpoint
3. Print evaluation metrics
4. Generate visualization plots

---

## Key Takeaways

* Neural networks can **learn to prune themselves during training**.
* Adding a **sparsity penalty** encourages unnecessary connections to disappear.
* There is a **trade-off between accuracy and sparsity**.
* Proper engineering practices (custom layers, experiments, analysis) are critical for building efficient AI systems.

---

## Future Improvements

Possible extensions include:

* Structured pruning (neuron-level pruning)
* Converting the pruned network into a **smaller architecture**
* Applying the technique to **CNN architectures**
* Exploring **Hard-Concrete gates for true binary pruning**

---

## Author

AI Engineering Case Study Implementation
Self-Pruning Neural Network for Efficient Deep Learning
