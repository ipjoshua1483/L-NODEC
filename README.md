# Lyapunov Neural Ordinary Differential Equation Feedback Control Policies

This repository contains the implementation of the Lyapunov Neural Ordinary Differential Equation Feedback Control Policies (LNODEC) in PyTorch. 

This paper presents a Lyapunov approach to Neural ODEs for solving continuous-time optimal control problems, for stabilizing a known constrained nonlinear system around a desired equilibrium state. We show that state feedback control policies learned via the LNODEC strategy leads to exponential stability of the controlled system and adversarial robustness to uncertain initial conditions. This is highlighted through two case studies: double integrator and optimal control for thermal dose delivery via a cold atmospheric plasma biomedical system. 

## Installation
Install Python 3.10.16

Run the following command: ```pip install -r requirements.txt```
<!--
## Double Integrator
![Double Integrator Phase Portrait](./figures/double_integrator_phase_portrait_subplots_labels.png)
*Phase portraits of the controlled double integrator system. Left: State trajectories for NODEC (no Lyapunov formulation). Right: State trajectories for LNODEC. Blue trajectories signify streamlines in the phase space. The nominal trajectory is shown in red and the adversarial trajectories with respect to perturbations in initial states are shown in orange.*

The nominal trajectories demonstrate that the LNDOEC formulation reaches equilibrium due to a zero terminal velocity, whereas the NODEC strategy does not. The adversarial trajectories also verify that LNODEC is more robust to uncertain initial conditions since all trajectories reach the desired terminal state, whereas the same cannot be said for NODEC.

## Atmospheric Pressure Plasma Jet (APPJ)
![APPJ CEM T u vs t](./figures/appj_NODEC_LNODEC_x_t_subplots_labels.png)
*Optimal control of thermal dose delivery of cold atmospheric plasma to a target surface. (a) The delivered thermal dose CEM. (b) Surface
temperature. (c) A sample control input, i.e., applied power to plasma, designed by NODEC and L-NODEC. 20 state profiles are shown based
on repeated training of the state-feedback neural control policy for each strategy. Note that the plasma treatment is terminated once the desired
terminal thermal dose is reached and, thus, the trajectories are truncated when CEM reaches 1.5 min.*

Across all 20 trajectories, the LNODEC strategy requires only at most 60s to reach the desired CEM whereas NODEC requires the full 100s, exhibiting a 40% improvement, which is highly desirable in the context where potential patients are involved. The LNODEC trajectories are also more consistent given that the NODEC trajectories experience significant variance in the intermediate time steps.
-->
