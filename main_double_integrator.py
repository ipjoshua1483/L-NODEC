import numpy as np
import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from datetime import datetime

from policy import Policy
from utils import (dtype, device)
# from case_study import Case_Study
from double_integrator import double_integrator_lya_obj, double_integrator_lya_obj_constrained

tkwargs = {
    "dtype": dtype,
    "device": device,
}

input_size = 2
hidden_size = 32
output_size = 1
policy_output_lower_bound = -10.
policy_output_upper_bound = 10.

def main():
    timestamp = datetime.now()
    timestamp_name = timestamp.strftime("%Y%m%d %H%M%S")

    node_policy = Policy(input_size, hidden_size, output_size, (policy_output_lower_bound, policy_output_upper_bound)).to(**tkwargs)
    lya_policy = Policy(input_size, hidden_size, output_size, (policy_output_lower_bound, policy_output_upper_bound)).to(**tkwargs)
    lya_constrained_policy = Policy(input_size, hidden_size, output_size, (policy_output_lower_bound, policy_output_upper_bound)).to(**tkwargs)
    
    print("Training NODE policy with no velocity reference")
    node_policy, node_times, node_states, node_inputs, node_objectives, node_loss_history = node_policy.train_policy(double_integrator_lya_obj)
    node_adversarial_time, node_adversarial_trajectories, node_adversarial_inputs, node_adversarial_terminal_mean, node_adversarial_terminal_std, node_adversarial_constraint_violations = node_policy.generate_adversarial_trajectories(double_integrator_lya_obj, node = True)
    node_vector_field_args = node_policy.generate_vector_fields(double_integrator_lya_obj)
    
    print("Training lyapunov policy")
    lya_policy, lya_times, lya_states, lya_inputs, lya_pointwise_losses, lya_potential_norm, lya_total_loss_history, lya_control_loss_history = lya_policy.train_policy(double_integrator_lya_obj, node = False)
    lya_adversarial_time, lya_adversarial_trajectories, lya_adversarial_inputs, lya_adversarial_terminal_mean, lya_adversarial_terminal_std, lya_adversarial_constraint_violations, lya_adversarial_potential_norm = lya_policy.generate_adversarial_trajectories(double_integrator_lya_obj)
    lya_vector_field_args = lya_policy.generate_vector_fields(double_integrator_lya_obj)

    print("Training lyapunov constrained policy")
    lya_constrained_policy, lya_constrained_times, lya_constrained_states, lya_constrained_inputs, lya_constrained_pointwise_losses, lya_constrained_potential_norm, lya_constrained_total_loss_history, lya_constrained_control_loss_history = lya_policy.train_policy(
        double_integrator_lya_obj_constrained, 
        node = False, 
        num_epochs = 400
    )
    lya_constrained_adversarial_time, lya_constrained_adversarial_trajectories, lya_constrained_adversarial_inputs, lya_constrained_adversarial_terminal_mean, lya_constrained_adversarial_std, lya_constrained_adversarial_constraint_violations, lya_constrained_adversarial_potential_norm = lya_constrained_policy.generate_adversarial_trajectories(double_integrator_lya_obj_constrained)

if __name__ == "__main__":
    main()