import numpy as np
import torch
# from torchdiffeq import odeint
import matplotlib.pyplot as plt
from datetime import datetime

from policy_2 import Policy
from utils import device
from appj import appj_lya_obj as obj

hidden_size = 32

def main():
    timestamp = datetime.now()
    timestamp_name = timestamp.strftime("%Y%m%d %H%M%S")

    node_policy = Policy(len(obj.init_cond), hidden_size, len(obj.u_bounds[0]), (obj.u_bounds[0], obj.u_bounds[1]), obj.normalize, obj.unnormalize).to(device)
    lya_policy = Policy(len(obj.init_cond), hidden_size, len(obj.u_bounds[0]), (obj.u_bounds[0], obj.u_bounds[1]), obj.normalize, obj.unnormalize).to(device)
    print("Training NODE policy with no velocity reference")
    node_policy_dynamics = node_policy.Dynamics(node_policy, obj.dynamics)
    node_policy, node_times, node_states, node_inputs, node_objectives, node_loss_history = node_policy_dynamics.train_policy(obj)
    
    print("Training lyapunov policy")
    lya_policy_dynamics = lya_policy.Dynamics(lya_policy, obj.dynamics)
    lya_policy, lya_times, lya_states, lya_inputs, lya_pointwise_losses, lya_potential_norm, lya_total_loss_history, lya_control_loss_history = lya_policy_dynamics.train_policy(obj, node = False)

if __name__ == "__main__":
    main()