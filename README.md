# CIFAR-10 Classification with Hybrid CNN

**Author:** Ritwik Jain

[![Accuracy](https://img.shields.io/badge/Accuracy-93.67%25-success)](#)
[![Parameters](https://img.shields.io/badge/Parameters-468k-blue)](#)

## 1. Introduction
This project features a high-performance Convolutional Neural Network (CNN) developed for the CIFAR-10 image classification task. The primary constraint was a strict parameter budget of under 500,000 parameters, necessitating a highly efficient architecture. The core objective was to maximize test accuracy without exceeding this limit, culminating in a model that achieves **93.67% accuracy** using exactly **468,000 parameters**.

## 2. Model Architecture
To balance the trade-off between model depth, width, and parameter cost, this project utilizes a custom 6-layer hybrid CNN architecture. The design transitions from standard convolutions in the early layers to depthwise separable convolutions in the deeper layers.

* **Total Parameters:** 468,000
* **Input:** 32x32 RGB Images
* **Output:** 10 Class Logits

### 2.1 Layer-by-Layer Breakdown

| Stage | Layer Type | Rationale |
| :--- | :--- | :--- |
| **Input** | Standard Conv | Immediate expansion to capture low-level features (edges, colors). |
| **Block 1** | Standard Conv + MaxPool | Refinement of low-level features using high-quality dense kernels. Reduces spatial dimensions from 32x32 to 16x16. |
| **Block 2** | Standard Conv + MaxPool | Increasing channel depth to capture textures and shapes. Reduces spatial dimensions from 16x16 to 8x8. |
| **Block 3** | Depthwise Separable Conv | Critical Step: Massive width expansion. Using DS-Conv reduces computational cost by ~85%, allowing for 384 channels. |
| **Block 4** | Depthwise Separable Conv + MaxPool| Deep semantic reasoning in high-dimensional space. Reduces spatial dimensions from 8x8 to 4x4. |
| **Head** | Global Average Pooling (GAP) | Flattens spatial data. Drastically reduces parameters compared to a standard Flatten + Fully Connected approach. |
| **Output** | Linear (FC) | Final classification scores. |

### 2.2 Justification of Design Choices

1.  **SiLU (Swish) Activation:** Standard ReLU was replaced with SiLU (x * sigmoid(x)). In plain networks without skip connections, deep layers often suffer from vanishing gradients. SiLU's smooth, non-monotonic nature allows for better gradient flow, effectively acting as a "soft" residual connection.
2.  **Depthwise Separable Convolutions:** Standard convolutions at 384 channels would require millions of parameters. DS-Convs split the operation into spatial learning and channel mixing, costing only ~160k parameters while maintaining similar representational power. 
3.  **Global Average Pooling (GAP):** Instead of flattening the feature map directly (which creates a massive fully connected layer), GAP is utilized. This significantly reduces the parameter count and makes the model invariant to small spatial shifts.

## 3. Training Strategy
The training pipeline was engineered to combat overfitting and maximize generalization.

### 3.1 Data Augmentation Strategy
An aggressive augmentation strategy was employed to force the model to learn robust features rather than memorizing pixels:
* Random Horizontal Flip & Crop
* Color Jitter
* Random Erasing (Cutout)

### 3.2 Optimization & Scheduler
* **Optimizer:** SGD with Momentum (0.9). SGD generalizes better than Adam for image classification tasks by finding "flatter" minima in the loss landscape.
* **Scheduler:** OneCycleLR. This technique starts with a low learning rate, ramps up to a high peak (`max_lr=0.05`) to traverse the loss landscape quickly, and then decays to near-zero. This "Super Convergence" phenomenon enables high accuracy in fewer epochs.
* **Label Smoothing:** Set to 0.1 in the `CrossEntropyLoss` function. This prevents the model from becoming overconfident (predicting 100% probability), improving generalization on unseen test data.

### 3.3 Inference Enhancement
* **Test Time Augmentation (TTA):** During evaluation, both the original image and a horizontally flipped version are passed through the network. The predictions are averaged, effectively creating an ensemble of two models at zero training cost. This typically boosts accuracy by ~1.5%.

## 4. Iteration History
The final architecture was the result of rigorous iterative testing and bottleneck analysis.

| Iteration | Architecture | Key Changes | Observation |
| :--- | :--- | :--- | :--- |
| **v1.0** | 4-Layer Plain CNN | Basic Augmentation, Constant LR | Hit a ceiling. Model was under-fitting due to shallow depth. |
| **v2.0** | 4-Layer CNN | Added OneCycleLR Scheduler | Better optimization improved results, but capacity was too low. |
| **v3.0** | 4-Layer CNN + Drop | Added Cutout + Dropout | Regularization helped, but plain network depth limited feature learning. |
| **v4.0** | DeepNet (Hybrid) | SiLU + Width=384 + Label Smooth | Massive improvement. Width expansion and SiLU solved the bottleneck. |
| **v5.0** | DeepNet + TTA | TTA Enabled | TTA created a zero-cost ensemble, pushing final accuracy to **93.67%**. |

## 5. Conclusion
By carefully managing the parameter budget using Depthwise Separable Convolutions and employing modern training techniques like OneCycleLR, Cutout, and SiLU activations, this project successfully delivers a high-performance model under the 500k parameter limit. The transition from a standard 4-layer CNN to the final "DeepNet-Max" architecture demonstrates the critical importance of maximizing model width within strict constraints and utilizing advanced regularization to prevent overfitting.
