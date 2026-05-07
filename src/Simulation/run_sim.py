"""
UCLARC: Nicolin Govender
5/5/26


--------------------------
Integrates existing  logic (Tasks/Agents) with GPU Physics (CUDA Warp)
As a mock it reads starting conditions from AgentsLog.csv, calculates HashGrid transmission,
and executes BVH spatial queries.
"""
import numpy as np
import warp as wp
import pandas as pd
import os
from enum import IntEnum

#==================================================================================================
# GPU Entry Point
wp.init()
#==================================================================================================


#==================================================================================================
class InfectionStatus(IntEnum):
    SUSCEPTIBLE = 0
    EXPOSED     = 1
    INFECTED    = 2
    RECOVERED   = 3
#==================================================================================================


#==================================================================================================
# 1] GPU Hash Grid for Agent-to-Agent Proximity or any other moving objects
#==================================================================================================
@wp.kernel
def cuda_warp_kernel_agent_proximity(
    grid: wp.uint64, # Spatial Grid

    positions: wp.array(dtype=wp.vec3),
    statuses: wp.array(dtype=wp.int32),
    floor_ids: wp.array(dtype=wp.int32),
    search_radius: float,
    out_infector: wp.array(dtype=wp.int32)
):
    tid = wp.tid() # Thread on GPU

    if statuses[tid] == int(0): # SUSCEPTIBLE
        pos = positions[tid]
        my_floor = floor_ids[tid]

        query = wp.hash_grid_query(grid, pos, search_radius)
        neighbor = int(0)

        while wp.hash_grid_query_next(query, neighbor):
            # Must be different agent, infected, and on the same floor since it is 2D ;-)
            if neighbor != tid and statuses[neighbor] == int(2) and floor_ids[neighbor] == my_floor:

                dist = wp.length(pos - positions[neighbor])
                if dist <= search_radius:
                    out_infector[tid] = neighbor
                    break
#==================================================================================================


#==================================================================================================
# 2] GPU BVH distance between any moving object and the static geometry
#==================================================================================================
@wp.kernel
def cuda_warp_kernel_geometry_all_agents(
    mesh: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    out_dist: wp.array(dtype=float)
):
    tid = wp.tid()

    # BVH Query per agent 15.0 units away
    hit = wp.mesh_query_point(mesh, positions[tid], 15.0, float(0.0), int(0), float(0.0), float(0.0))
    if hit:
        closest_pt = wp.mesh_eval_position(mesh, int(0), float(0.0), float(0.0))
        out_dist[tid] = wp.length(positions[tid] - closest_pt)
    else:
        out_dist[tid] = -1.0
#==================================================================================================


#==================================================================================================
# 3] Main to run the simulation
#==================================================================================================
def process_and_simulate():
    print("Starting Wrap Simulation")

    # A. Load the correctly formatted CSV
    try:
        df = pd.read_csv("../../Data/agents_log.csv")
        geom = np.load("../../Data/floorplan_simple_a.npz", allow_pickle=True)
    except FileNotFoundError:
        print("❌ Error: Missing agents_log.csv or NPZ file.")
        return

    # Add a dummy Z-axis for Warp's vec3 requirement
    df['pos_z'] = 1.0

    times = sorted(df['time'].unique())
    grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128)
    search_radius = 2.0
    transmission_ledger = []

    # Map status dictionary using your new Enum values
    agent_status_map = {aid: InfectionStatus.SUSCEPTIBLE.value for aid in df['agent_id'].unique()}

    # Pre-seed Patient Zero if they exist in the initial frame
    initial_infected = df[(df['time'] == times[0]) & (df['infection_status'] == InfectionStatus.INFECTED.value)]
    for _, row in initial_infected.iterrows():
        agent_status_map[row['agent_id']] = InfectionStatus.INFECTED.value


    #-------------------------------------------------------------
    # Time step loop
    #-------------------------------------------------------------
    for t in times:

        #-------------------------------------------------------------
        # 1. MOCKA: For this script, we read the intended targets/positions from the CSV
        frame_mask = df['time'] == t
        frame_df = df[frame_mask].sort_values('agent_id').copy()
        # Sync GPU-calculated infection states back to the Python DataFrame representation
        frame_df['infection_status'] = frame_df['agent_id'].map(agent_status_map)
        #-------------------------------------------------------------

        #-------------------------------------------------------------
        # 2. Memory Transfer (Python -> GPU)
        pos_np = frame_df[['pos_x', 'pos_y', 'pos_z']].to_numpy(dtype=np.float32)
        status_np = frame_df['infection_status'].to_numpy(dtype=np.int32)
        floor_np = frame_df['floor'].to_numpy(dtype=np.int32)
        agent_ids_np = frame_df['agent_id'].to_numpy(dtype=np.int32)

        wp_pos = wp.array(pos_np, dtype=wp.vec3)
        wp_status = wp.array(status_np, dtype=wp.int32)
        wp_floor = wp.array(floor_np, dtype=wp.int32)
        wp_infector_idx = wp.full(len(pos_np), -1, dtype=wp.int32)
        #-------------------------------------------------------------

        #-------------------------------------------------------------
        # 3. GPU does the spatial query
        #-------------------------------------------------------------
        # 3.1. Build Spatial Grid
        grid.build(points=wp_pos, radius=search_radius)

        # 3.2 Calculate Proximity and Transmissions
        wp.launch(
            kernel=cuda_warp_kernel_agent_proximity,
            dim=len(pos_np),
            inputs=[grid.id, wp_pos, wp_status, wp_floor, search_radius, wp_infector_idx]
        )
        wp.synchronize()
        #-------------------------------------------------------------

        #-------------------------------------------------------------
        # 4. Memory Transfer (GPU -> Python)
        #-------------------------------------------------------------
        infector_idx_np = wp_infector_idx.numpy()
        for i, inf_idx in enumerate(infector_idx_np):
            if inf_idx != -1:
                infectee_id = agent_ids_np[i]
                infector_id = agent_ids_np[inf_idx]

                if agent_status_map[infectee_id] == InfectionStatus.SUSCEPTIBLE.value:
                    # Update Python State
                    agent_status_map[infectee_id] = InfectionStatus.INFECTED.value

                    infectee_row = frame_df.iloc[i]
                    infector_row = frame_df.iloc[inf_idx]
                    dist = np.linalg.norm([infectee_row['pos_x'] - infector_row['pos_x'],
                                           infectee_row['pos_y'] - infector_row['pos_y']])

                    print(f" TRANSMISSION DETECTED: Agent {infector_id} -> Agent {infectee_id} at t={t} (Dist: {dist:.2f}m)")

                    transmission_ledger.append({
                        'time': t, 'infector': infector_id, 'infectee': infectee_id,
                        'building_idx': infectee_row['building_idx'], 'floor': infectee_row['floor'],
                        'pos_x': infectee_row['pos_x'], 'pos_y': infectee_row['pos_y'], 'dist': dist
                    })
        #-------------------------------------------------------------

        # Apply the globally updated statuses back to the main DataFrame
        df.loc[frame_mask, 'infection_status'] = df.loc[frame_mask, 'agent_id'].map(agent_status_map)

    #-------------------------------------------------------------
    # End Time step loop
    #-------------------------------------------------------------

    # Save the Transmission Ledger
    pd.DataFrame(transmission_ledger).to_csv("../../Results/transmission_events.csv", index=False)

    # Export Final Processed Data
    out_file = "../../Results/processed_telemetry.csv"
    df.to_csv(out_file, index=False)
    print(f"Simulation Complete. Telemetry saved to {out_file}")
#==================================================================================================


#==================================================================================================
if __name__ == "__main__":
    process_and_simulate()
#==================================================================================================




#==================================================================================================
# Start ML Bridge Template @Edil Direct Memory sharing
#==================================================================================================
import torch
import torch.nn as nn

#=====================================================================
class DummyPolicyNetwork(nn.Module):

    #-------------------------------------------------------------
    """ 1. A simple mock Neural Network to representRL Policy."""
    def __init__(self):
        super().__init__()
        # Input: 3D position + 1D status = 4 features
        # Output: 2D target (x, y destination)
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    #-------------------------------------------------------------

    #-------------------------------------------------------------
    # 2. Simple mock update
    def forward(self, positions, statuses):
        # Concatenate position and status for the network
        x = torch.cat([positions, statuses.unsqueeze(1).float()], dim=1)
        return self.net(x)
    #-------------------------------------------------------------

#=====================================================================

#=====================================================================
# Initialize the global network (push to CUDA)
rl_policy_net = DummyPolicyNetwork().cuda()
#=====================================================================


#=====================================================================
def execute_neural_policy(wp_positions, wp_statuses):
    """
    Zero-Copy bridge from Warp to PyTorch
    This will replace MOCKA which is read as a CSV)
    """
    # 1. Map GPU DRAM directly to PyTorch
    tensor_pos = wp.to_torch(wp_positions)
    tensor_status = wp.to_torch(wp_statuses)

    # 2. Feed to Neural Network to get new destinations
    with torch.no_grad():
        predicted_targets = rl_policy_net(tensor_pos, tensor_status)

    # 3. Map the PyTorch output back to a Warp array for the solver
    wp_new_targets = wp.from_torch(predicted_targets, dtype=wp.vec2)

    return wp_new_targets
#=====================================================================


#=====================================================================
# Warp Kernels using Auto Diff
#=====================================================================


#=====================================================================
@wp.kernel
def cuda_warp_kernel_continuous_exposure(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    viral_loads: wp.array(dtype=float),
    transmission_rate: wp.array(dtype=float), # What we want to solve
    search_radius: float,
    out_exposure: wp.array(dtype=float)
):
    tid = wp.tid()

    pos = positions[tid]
    query = wp.hash_grid_query(grid, pos, search_radius)
    neighbor = int(0)

    # Calculate cumulative viral exposure dynamically
    while wp.hash_grid_query_next(query, neighbor):
        if neighbor != tid:
            dist = wp.length(pos - positions[neighbor])
            if dist <= search_radius and dist > 0.1: # Prevent divide-by-zero because python is not C ;-)
                # Physics Equation: Exposure = (1 / dist^2) * load * rate
                exposure_dose = (1.0 / (dist * dist)) * viral_loads[neighbor] * transmission_rate[0]
                wp.atomic_add(out_exposure, tid, exposure_dose)

#=====================================================================


#=====================================================================
def calibrate_disease_parameters(wp_pos, grid, num_agents):
    """
    Uses Warp's AutoDiff to find the real-world transmission rate
    by matching simulation output to real-world target data using Gradients
    """
    print("\nStart Differentiable Parameter Calibration using Warp ")

    # 1. Set up dummy viral loads
    np_viral = np.zeros(num_agents, dtype=np.float32)
    np_viral[0] = 1.0 # Patient Zero is shedding virus
    wp_viral_loads = wp.array(np_viral, dtype=float)

    # 2. This is the Real Wold data we want to match (e.g. Sensors/CheckPoints/Wearables)
    np_target_exposure = np.zeros(num_agents, dtype=np.float32)
    np_target_exposure[5] = 0.85 # eg Agent 5 got sick with an 0.85 dose
    wp_target = wp.array(np_target_exposure, dtype=float, requires_grad=False)

    # 3. The Parameter we are trying to guess keep withing FP8 Limits
    # requires_grad=True tells the GPU to track the mathematical derivative
    guessed_rate = wp.array([0.1], dtype=float, requires_grad=True)
    out_exposure = wp.zeros(num_agents, dtype=float, requires_grad=True)

    learning_rate = 0.05

    # 4. Gradient Descent Loop
    for step in range(50):

        # Clear previous gradients
        guessed_rate.grad.zero_()
        out_exposure.zero_()

        # Record the forward simulation pass
        tape = wp.Tape()
        with tape:
            wp.launch(
                kernel=cuda_warp_kernel_continuous_exposure,
                dim=num_agents,
                inputs=[grid.id, wp_pos, wp_viral_loads, guessed_rate, 2.0, out_exposure]
            )
            # Calculate error against real-world data
            loss = wp.optim.mse(out_exposure, wp_target)

        # Execute the backward pass to compute gradients
        tape.backward(loss)

        # Update our guessed parameter using the calculated gradient
        rate_grad = guessed_rate.grad.numpy()[0]
        current_rate = guessed_rate.numpy()[0]

        # Gradient Descent Step
        new_rate = current_rate - (learning_rate * rate_grad)
        guessed_rate = wp.array([new_rate], dtype=float, requires_grad=True)

        if step % 10 == 0:
            print(f"Step {step:02d} | Loss: {loss.numpy()[0]:.6f} | Guessed Trans_Rate: {current_rate:.4f} | Grad: {rate_grad:.4f}")

    print(f" Calibration Complete. Inferred Transmission Rate: {guessed_rate.numpy()[0]:.4f}\n")
#=====================================================================

#==================================================================================================
# End ML Bridge Template
#==================================================================================================

