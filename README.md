# Neural-Ordinary-Differential-Equations
## A Reduced-Scale Implementation and Experimental Study

A PyTorch-based implementation and experimental study of **Neural Ordinary Differential Equations (Neural ODEs)**, based on the paper by Chen et al. (NeurIPS 2018).

This project explores continuous-depth neural networks, ODE-based dynamics, adjoint sensitivity, numerical integration, and latent ODEs through reduced-scale experiments.

## Overview

Traditional neural networks process data through a sequence of discrete layers. Neural ODEs instead define the hidden state as the solution of an ordinary differential equation:

\[
\frac{dh(t)}{dt}=f_\theta(h(t),t)
\]

where \(h(t)\) is the hidden state and \(f_\theta\) is a neural network parameterized by \(\theta\).

An ODE solver computes the hidden state over a specified time interval. The model can be trained using standard automatic differentiation or the adjoint sensitivity method.

### Objectives

- Implement Neural ODE models using PyTorch.
- Study direct differentiation and adjoint-based gradient computation.
- Investigate numerical integration accuracy using different RK4 step sizes.
- Fit a continuous dynamical system and evaluate latent ODE behavior.
- Compare Neural ODEs with ResNet baselines on classification tasks.
- Conduct reduced-scale experiments and analyze parameter efficiency, accuracy, and runtime.

## Repository Structure

```text
Neural-Ordinary-Differential-Equations/
├── README.md
├── Neural_ODE_Results_Verified.ipynb
├── Neural_ODE_IEEE_Report_Updated.docx
└── Neural_ODE_Presentation.pptx
```

## Experiments and Results

The following results are from the saved experimental notebook. Runtime measurements depend on the execution environment and should not be interpreted as hardware-independent benchmarks.

### 1. Spiral Classification

| Model | Test Accuracy | Parameters | Runtime |
|---|---:|---:|---:|
| Neural ODE | 71.88% | 8,642 | 24.62 s |
| ResNet | 69.38% | 33,602 | 2.89 s |

**Observation:** Neural ODE achieved a 2.50 percentage-point accuracy improvement with approximately 74.3% fewer parameters. However, it required more computation time in this experiment.

### 2. Direct Differentiation vs. Adjoint Method

| Metric | `odeint` | `odeint_adjoint` |
|---|---:|---:|
| Loss | 0.663166 | 0.663166 |
| Gradient Norm | 0.645644 | 0.645644 |
| Runtime | 0.0230 s | 0.1159 s |

**Observation:** Both methods produced matching loss and gradient-norm values in this small experiment. The adjoint implementation was slower for this particular run. A single timing measurement on a small workload does not establish general performance or memory advantages.

### 3. Custom Adjoint Gradient Verification

A scalar differential equation was used to compare automatic differentiation against a custom adjoint calculation.

| Metric | Result |
|---|---:|
| Terminal State | 1.702881 |
| Loss | 0.04413983 |
| Autograd Gradient | -0.50595822 |
| Custom Adjoint Gradient | -0.50595822 |
| Absolute Gradient Difference | \(1.110 \times 10^{-16}\) |

**Observation:** The gradients agree to numerical precision for this toy problem. The custom calculation uses trapezoidal integration for the parameter-gradient integral and is a special-case demonstration, not a general-purpose adjoint solver.

### 4. Continuous Dynamical System Fitting

A damped-rotation system was used to evaluate how well a learned differential equation reproduces continuous dynamics.

| Metric | Result |
|---|---:|
| Derivative MSE at Epoch 0 | 1.808793 |
| Derivative MSE at Epoch 300 | 0.000087 |
| Trajectory MSE | 0.000006 |

**Observation:** The reduction in derivative and trajectory errors indicates that the learned dynamics closely approximate the target system in this experiment.

### 5. RK4 Step-Size Ablation

| RK4 Step Size | MSE | Runtime |
|---:|---:|---:|
| 0.20 | 0.0000857 | 0.0332 s |
| 0.10 | 0.0000102 | 0.0434 s |
| 0.05 | 0.0000056 | 0.0667 s |

**Observation:** Smaller integration steps reduced numerical error but increased runtime, illustrating the accuracy–computation trade-off in numerical ODE solving.

### 6. Latent ODE Beta Ablation

The KL-divergence coefficient \(\beta\) was varied to investigate its effect on reconstruction quality and latent regularization.

| Beta | Clean MSE | Noisy MSE | KL Divergence | Runtime |
|---:|---:|---:|---:|---:|
| 0.001 | 0.025113 | 0.025628 | 2.0222 | 15.5 s |
| 0.0001 | 0.026189 | 0.026697 | 2.2602 | 15.5 s |

**Observation:** In these runs, \(\beta=0.001\) achieved lower clean and noisy reconstruction errors. The results illustrate how the KL regularization weight can influence reconstruction and latent-space behavior.

### 7. MNIST Classification

The MNIST experiment used 20,000 training examples, 5,000 test examples, and five training epochs.

| Model | Test Accuracy | Test Error | Parameters | Runtime |
|---|---:|---:|---:|---:|
| Neural ODE | 92.96% | 7.04% | 22,026 | 39.62 s |
| ResNet | 93.64% | 6.36% | 46,986 | 28.20 s |

**Observation:** ResNet achieved 0.68 percentage points higher accuracy, while Neural ODE used approximately 53.1% fewer parameters. Neural ODE took longer in this experiment.

## Comparison with the Original Paper

The original Neural ODE paper reports the following results for its image-classification experiments:

| Model | Test Error | Parameters |
|---|---:|---:|
| ResNet | 0.41% | 0.60M |
| RK-Net | 0.47% | 0.22M |
| ODE-Net | 0.42% | 0.22M |

These results are included as context from the original paper. They are **not a direct, apples-to-apples comparison** with this project because the datasets, architectures, training configurations, and experimental scales may differ.

## Technology Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **ODE Solvers:** `torchdiffeq`
- **Experiments:** Jupyter Notebook / Google Colab
- **Models:** Neural ODE, ResNet, latent ODE components
- **Numerical Integration:** Runge–Kutta 4 (RK4)

## Installation

Clone the repository:

```bash
git clone https://github.com/satyasheel2406/Neural-Ordinary-Differential-Equations.git
cd Neural-Ordinary-Differential-Equations
```

Create and activate a virtual environment if desired:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install torch torchvision torchdiffeq numpy matplotlib scipy scikit-learn jupyter
```

Launch the notebook:

```bash
jupyter notebook Neural_ODE_Results_Verified.ipynb
```

Alternatively, upload the notebook to Google Colab, install the dependencies in a notebook cell, and execute the experiment cells in order.

## Reproducibility

- Use the saved notebook as the entry point for reviewing the experiments.
- Set the random seeds in the notebook before rerunning experiments.
- Keep the dataset splits, model configurations, training epochs, and solver settings unchanged when reproducing the reported results.
- Runtime may vary with CPU/GPU availability, hardware, software versions, and solver behavior.
- The reported values are saved experimental results; the corrected results table itself does not establish a fresh, clean, end-to-end rerun in a new environment.

## Limitations

- Experiments use reduced-scale configurations and are intended for educational and comparative analysis.
- Runtime comparisons are based on individual recorded runs and are not statistically averaged benchmarks.
- The scalar custom adjoint implementation demonstrates a special case rather than a general Neural ODE training algorithm.
- Repeated test-set evaluation during training can introduce evaluation bias; a separate validation set is preferable for model selection.
- Results should not be directly compared with the original paper without matching its datasets, architectures, training procedures, and evaluation protocols.

## Key Takeaways

1. Neural ODEs represent hidden-state evolution using continuous-time dynamics.
2. Adjoint sensitivity provides an alternative approach to gradient computation, but its practical speed depends on the workload and implementation.
3. Smaller RK4 step sizes improved numerical accuracy while increasing runtime in the tested system.
4. Neural ODEs used fewer parameters than the ResNet baselines in both classification experiments.
5. Parameter efficiency did not consistently translate into higher accuracy or faster execution.

## Reference

Chen, R. T. Q., Rubanova, Y., Bettencourt, J., and Duvenaud, D. (2018). **Neural Ordinary Differential Equations.** Advances in Neural Information Processing Systems (NeurIPS).

Paper: https://arxiv.org/abs/1806.07366

---

**Project:** Neural Ordinary Differential Equations — A Reduced-Scale Implementation and Experimental Study  
**Author:** Satyasheel Raman  
**Roll Number:** 24CS3047  
**Institute:** Rajiv Gandhi Institute of Petroleum Technology (RGIPT)
