# Perceptron

## Introduction

The **Perceptron** is one of the earliest and simplest machine learning algorithms used for **binary classification**. It is a type of **single-layer artificial neural network** that learns a **linear decision boundary** between classes.

The perceptron was introduced by Frank Rosenblatt in 1958 and became the foundation of modern neural networks and deep learning. ([digitallibrary.mes.ac.in][1])

The perceptron works by computing a weighted sum of input features and applying an activation function to produce an output.

## Basic Idea

The perceptron mimics a biological neuron:

* Inputs → Features
* Weights → Synaptic strengths
* Summation → Neuron body
* Activation → Output signal

## Perceptron Model Diagram

```
         x1 ----(w1)---
                       \
        x2 ----(w2)---->   Σ (Weighted Sum) ---> Activation ---> Output (y)
                       /
         x3 ----(w3)---

                 + Bias (b)
```

Or mathematically:

```
y = f(w1x1 + w2x2 + w3x3 + ... + wn xn + b)
```


## Mathematical Representation

### Weighted Sum

[
z = w_1x_1 + w_2x_2 + w_3x_3 + ... + w_nx_n + b
]

or in vector form:

[
z = w^T x + b
]

Where:

* (x) = Input vector
* (w) = Weight vector
* (b) = Bias
* (z) = Linear combination


### Activation Function

The perceptron uses a **step function**:

[
y =
\begin{cases}
1 & \text{if } z \geq 0 \
0 & \text{if } z < 0
\end{cases}
]

## Learning Rule

The perceptron updates weights using the rule:

[
w = w + \eta(y_{true} - y_{pred})x
]

[
b = b + \eta(y_{true} - y_{pred})
]

Where:

* ( \eta ) = Learning rate
* ( y_{true} ) = Actual label
* ( y_{pred} ) = Predicted label

## Training Algorithm

### Step 1

Initialize weights randomly.

### Step 2

For each training example:

1. Compute output

[
y = f(w^Tx + b)
]

2. Update weights

[
w = w + \eta(y_{true} - y)x
]

3. Update bias

[
b = b + \eta(y_{true} - y)
]

### Step 3

Repeat until convergence.


## Decision Boundary

The perceptron learns a **linear boundary**:

### 2D Example:

[
w_1x_1 + w_2x_2 + b = 0
]

This equation represents a straight line separating two classes.

## Advantages

* Simple and fast
* Easy to implement
* Works well for linearly separable data
* Foundation of neural networks

## Limitations

* Cannot solve non-linear problems (e.g., XOR)
* Only binary classification
* Sensitive to feature scaling


## Applications

* Spam detection
* Text classification
* Image classification (basic)
* Pattern recognition


## Visualization

### Linear Decision Boundary

```
Class 1  ● ● ● ● ●

--------------------  ← Decision Boundary

Class 0  ○ ○ ○ ○ ○
```

## Historical Background

The perceptron was inspired by early artificial neuron models such as the McCulloch–Pitts neuron proposed in 1943. ([Wikipedia][2])

It was the **first algorithm capable of learning weights automatically**, making it a milestone in artificial intelligence.

## Key Research Papers

### 1. Original Perceptron Paper

**Frank Rosenblatt (1958)**
"The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"

Paper Link:
[https://digitallibrary.mes.ac.in/handle/1/7167](https://digitallibrary.mes.ac.in/handle/1/7167)

This is the original perceptron paper that introduced the algorithm. ([digitallibrary.mes.ac.in][1])


### 2. First Artificial Neuron Paper

**McCulloch & Pitts (1943)**
"A Logical Calculus of the Ideas Immanent in Nervous Activity"

Paper Link:
[https://en.wikipedia.org/wiki/A_Logical_Calculus_of_the_Ideas_Immanent_in_Nervous_Activity](https://en.wikipedia.org/wiki/A_Logical_Calculus_of_the_Ideas_Immanent_in_Nervous_Activity)

This paper introduced the first mathematical neuron model. ([Wikipedia][2])


### 3. Modern Perceptron Survey Paper

**Du et al. (2022)**
"Perceptron: Learning, Generalization, Model Selection, Fault Tolerance, and Role in the Deep Learning Era"

Paper Link:
[https://ouci.dntb.gov.ua/en/works/96V5kYo9/](https://ouci.dntb.gov.ua/en/works/96V5kYo9/)

Modern comprehensive survey of perceptron research. ([OUCI][3])


## Conclusion

The perceptron is the **foundation of modern neural networks** and deep learning. Although simple, it introduced key concepts such as:

* Weights
* Bias
* Learning rule
* Decision boundary

Modern deep learning architectures like multilayer perceptrons are direct extensions of this basic model.
