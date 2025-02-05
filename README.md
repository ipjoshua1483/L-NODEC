# Lyapunov Neural Ordinary Differential Equation State-Feedback Control Policies

This repository contains the implementation of the Lyapunov Neural Ordinary Differential Equation State-Feedback Control Policies (LNODEC) in PyTorch. 

This paper presents a Lyapunov approach to Neural ODEs for solving continuous-time optimal control problems, for stabilizing a known constrained nonlinear system around a desired equilibrium state. We show that state feedback control policies learned via the LNODEC strategy leads to exponential stability of the controlled system and adversarial robustness to uncertain initial conditions. This is highlighted through two case studies: double integrator and optimal control for thermal dose delivery via a cold atmospheric plasma biomedical system. 

## Installation
Install Python 3.10.16

Run the following command: ```pip install -r requirements.txt```

## Usage
To generate data for the first case study (double integrator), run the following command:

```python main_double_integrator.py```

To generate data for the second case study (atmospheric pressure plasma jet), run the following command:

```python main_appj.py```
