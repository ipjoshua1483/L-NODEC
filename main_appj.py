import numpy as np
import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

from policy_2 import Policy
from utils import (
    device, 
    plot_loss_epoch, 
    plot_x_t, 
    plot_u_t,
    plot_adversarial_appj, 
)
from appj import appj_lya_obj as obj
from appj import appj_terminal_obj

hidden_size = 32

plot_ref = torch.tensor([45, 1.5])

def main():
    timestamp = datetime.now()
    timestamp_name = timestamp.strftime("%Y%m%d %H%M%S")

    node_policy = Policy(len(obj.init_cond), hidden_size, len(obj.u_bounds[0]), (obj.u_bounds[0], obj.u_bounds[1]), obj.normalize, obj.unnormalize).to(device)
    node_terminal_policy = Policy(len(appj_terminal_obj.init_cond), hidden_size, len(appj_terminal_obj.u_bounds[0]), (appj_terminal_obj.u_bounds[0], appj_terminal_obj.u_bounds[1]), appj_terminal_obj.normalize, appj_terminal_obj.unnormalize).to(device)
    lya_policy = Policy(len(obj.init_cond), hidden_size, len(obj.u_bounds[0]), (obj.u_bounds[0], obj.u_bounds[1]), obj.normalize, obj.unnormalize).to(device)
    
    print("Training NODE policy with stage cost")
    node_policy_dynamics = node_policy.Dynamics(node_policy, obj.dynamics)
    node_policy, node_times, node_states, node_inputs, node_objectives, node_loss_history = node_policy_dynamics.train_policy(obj)
    node_adversarial_time, node_adversarial_trajectories, node_adversarial_inputs, node_adversarial_terminal_mean, node_adversarial_terminal_std, node_adversarial_constraint_violations = node_policy_dynamics.generate_adversarial_trajectories(
        obj, # node = True
    )
    
    print("Training NODE policy with terminal cost")
    node_terminal_policy_dynamics = node_terminal_policy.Dynamics(node_terminal_policy, appj_terminal_obj.dynamics)
    node_terminal_policy, node_terminal_times, node_terminal_states, node_terminal_inputs, node_terminal_objectives, node_terminal_loss_history = node_terminal_policy_dynamics.train_policy(appj_terminal_obj)
    node_terminal_adversarial_time, node_terminal_adversarial_trajectories, node_terminal_adversarial_inputs, node_terminal_adversarial_terminal_mean, node_terminal_adversarial_terminal_std, node_terminal_adversarial_constraint_violations = node_terminal_policy_dynamics.generate_adversarial_trajectories(
        appj_terminal_obj, # node = True
    )

    print("Training lyapunov policy")
    lya_policy_dynamics = lya_policy.Dynamics(lya_policy, obj.dynamics)
    lya_policy, lya_times, lya_states, lya_inputs, lya_pointwise_losses, lya_potential_norm, lya_total_loss_history, lya_control_loss_history = lya_policy_dynamics.train_policy(obj, node = False)
    lya_adversarial_time, lya_adversarial_trajectories, lya_adversarial_inputs, lya_adversarial_terminal_mean, lya_adversarial_terminal_std, lya_adversarial_constraint_violations = lya_policy_dynamics.generate_adversarial_trajectories(
        obj
    )

    data = {
        "node_stage_policy": [node_policy, node_policy_dynamics],
        "node_terminal_policy": [node_terminal_policy, node_terminal_policy_dynamics],
        "lya_policy": [lya_policy, lya_policy_dynamics],
        "node_stage_times": node_times,
        "node_terminal_times": node_terminal_times,
        "lya_times": lya_times,
        "node_stage_adversarial_trajectories": node_adversarial_trajectories,
        "node_stage_adversarial_inputs": node_adversarial_inputs,
        "node_terminal_adversarial_trajectories": node_terminal_adversarial_trajectories,
        "node_terminal_adversarial_inputs": node_terminal_adversarial_inputs,
        "lya_adversarial_trajectories": lya_adversarial_trajectories,
        "lya_adversarial_inputs": lya_adversarial_inputs,
    }
    with open(f"{timestamp_name} appj.pkl", "wb") as f:
        pickle.dump(data, f)
    print("Data saved successfully")
    # plot_adversarial_appj(
    #     [node_times, lya_times],
    #     [node_adversarial_trajectories, lya_adversarial_trajectories],
    #     plot_ref,
    #     ["NODEC", "L-NODEC"],
    #     ["blue", "green"],
    #     timestamp_name
    # )

    # loss_history_dict = {
    #     "NODE no vel ref": node_loss_history,
    #     "Lyapunov Control Loss": lya_control_loss_history,
    # }

    # plot_epoch = -1
    # #view loss histories
    # plot_loss_epoch(loss_history_dict, timestamp_name)

    # plot_epoch_min = torch.argmin(torch.tensor(lya_total_loss_history, dtype = torch.float32).to(device))
    # print(f"plot_epoch_min total loss: {plot_epoch_min}")
    # plot_epoch_min_2 = torch.argmin(torch.tensor(lya_control_loss_history, dtype = torch.float32).to(device))
    # print(f"plot_epoch_min control loss: {plot_epoch_min_2}")

    # for i in range(len(obj.init_cond)):
    #     ref = obj.x_ref[i]
    #     plot_x_t(i, ref, node_times, lya_times, node_states, lya_states, timestamp_name, plot_epoch)
    #     plot_x_t(i, ref, node_times, lya_times, node_states, lya_states, timestamp_name, plot_epoch_min)
    #     plot_x_t(i, ref, node_times, lya_times, node_states, lya_states, timestamp_name, plot_epoch_min_2)
        
    # plot_u_t(node_times, node_inputs, lya_inputs, timestamp_name, plot_epoch)
    # plot_u_t(node_times, node_inputs, lya_inputs, timestamp_name, plot_epoch_min)
    # plot_u_t(node_times, node_inputs, lya_inputs, timestamp_name, plot_epoch_min_2)

    # print("Plots saved successfully")

if __name__ == "__main__":
    main()