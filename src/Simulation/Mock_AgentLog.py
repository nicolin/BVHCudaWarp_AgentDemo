"""
UCLARC: Nicolin Govender
5/5/26

-------------------------------------------------
Generates curved walking paths, dynamic door events, and formats
output to match the existing Python Record class and Enums
"""

import pandas as pd
import numpy as np
import os
from enum import IntEnum

#==================================================================================================
class AgentType(IntEnum):
    GENERIC           = 0
    PATIENT           = 1
    HEALTHCARE_WORKER = 2
#==================================================================================================


#==================================================================================================
class InfectionStatus(IntEnum):
    SUSCEPTIBLE = 0
    EXPOSED     = 1
    INFECTED    = 2
    RECOVERED   = 3
#==================================================================================================


#=====================================================================
def generate_outbreak_data(timesteps=100):
    print(f"Generating 16-Agent Outbreak ({timesteps} timesteps)")
    agent_data = []

    #-----------------------------------------------------------
    # Mock Parameters and rules
    #-----------------------------------------------------------

    # Global Defaults
    building_idx = 0
    floor        = 0
    heading      = 0.0
    walk_speed_x = 0.3

    # Event Timings
    leave_bed_time    = 20
    travel_duration   = 8  # How many timesteps it takes to walk from bed to door
    reach_door_time   = leave_bed_time + travel_duration
    door_close_buffer = 7 # Keep door open for 7 ticks after passing


    # Spatial Bounds and Locations
    icu_bed_pos       = (2.0, 13.0)
    icu_center        = (4.0, 10.0)
    corridor_y_center = 3.0
    corridor_bounds_x = (1.0, 19.0)
    corridor_bounds_y = (1.0, 4.8)


    #-----------------------------------------------------------
    # Simple logic since the agent is not casper and cannot walk thru walls
    #-----------------------------------------------------------
    try:
        geom = np.load("../../Data/floorplan_simple_a.npz", allow_pickle=True)
        icu_door_pts = geom['doors'][0]
        door_x = (icu_door_pts[0][0] + icu_door_pts[1][0]) / 2.0
        door_y = (icu_door_pts[0][1] + icu_door_pts[1][1]) / 2.0
        print(f"Dynamic Waypoint found: Routing Agent 1 through Door at ({door_x:.2f}, {door_y:.2f})")
    except Exception as e:
        print("⚠️ Could not load CAD for routing. Defaulting door waypoint to (5.0, 5.0).")
        door_x, door_y = 5.0, 5.0

    #-----------------------------------------------------------
    # Time Loop
    #-----------------------------------------------------------
    for t in range(timesteps):

        #-----------------------------------------------------------
        # Agent 1 (Patient Zero): Bed -> Dynamic Waypoint -> Corridor
        #-----------------------------------------------------------
        if t < leave_bed_time:
            # Rule 1: Stay in bed until departure time
            x, y = icu_bed_pos

        elif t <= reach_door_time:
            # Rule 2: Linear interpolation vector from Bed to exact Door Waypoint
            progress = (t - leave_bed_time) / travel_duration
            x = icu_bed_pos[0] + ((door_x - icu_bed_pos[0]) * progress)
            y = icu_bed_pos[1] + ((door_y - icu_bed_pos[1]) * progress)

        else:
            # Rule 3: Enter Corridor, step down slightly, walk rightwards
            time_in_corridor = t - reach_door_time
            x = door_x + (time_in_corridor * walk_speed_x)
            y = (door_y - 1.0) + np.sin(t * 0.2) * 0.5

        agent_data.append([
            t, 1, building_idx, floor, x, y, heading,
            InfectionStatus.INFECTED.value, AgentType.PATIENT.value
        ])

        #-----------------------------------------------------------
        # Agents 2-4: ICU Staff (Pacing around the ICU Center)
        #-----------------------------------------------------------
        for i, offset in enumerate([0, 2, 4]):
            x = icu_center[0] + np.sin((t + offset) * 0.1) * 2
            y = icu_center[1] + np.cos((t + offset) * 0.15) * 3

            agent_data.append([
                t, i+2, building_idx, floor, x, y, heading,
                InfectionStatus.SUSCEPTIBLE.value, AgentType.HEALTHCARE_WORKER.value
            ])

        #-----------------------------------------------------------
        # Agents 5-16: Corridor Crowd (Spreading out within bounded rules)
        #-----------------------------------------------------------
        for i in range(5, 17):
            base_x = 2.0 + (i - 5) * 1.3
            speed_x = np.sin(t * 0.05 + i) * 1.5
            curve_y = np.cos(t * 0.1 + i) * 1.0

            # Rule 4: Clip agent positions strictly within corridor limits
            x = np.clip(base_x + speed_x, corridor_bounds_x[0], corridor_bounds_x[1])
            y = np.clip(corridor_y_center + curve_y, corridor_bounds_y[0], corridor_bounds_y[1])

            agent_data.append([
                t, i, building_idx, floor, x, y, heading,
                InfectionStatus.SUSCEPTIBLE.value, AgentType.GENERIC.value
            ])
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    # Save Agents Log
    #-----------------------------------------------------------
    columns = ['time', 'agent_id', 'building_idx', 'floor', 'pos_x', 'pos_y', 'heading', 'infection_status', 'agent_type']
    os.makedirs("../../Data", exist_ok=True)

    df_agents = pd.DataFrame(agent_data, columns=columns)
    df_agents.to_csv("../../Data/agents_log.csv", index=False)
    #-----------------------------------------------------------

    # -----------------------------------------------------------
    # Door Status: Synchronized with Agent 1 logic
    # -----------------------------------------------------------
    door_data = []

    for t in range(timesteps):
        # Rule 5: Door opens when agent leaves bed, closes after buffer period
        door1_open = 1 if leave_bed_time <= t <= (reach_door_time + door_close_buffer) else 0
        door_data.append([t, 1, door1_open])

        # Background ambient door opening for the Ward
        door2_open = 1 if (t // 15) % 2 == 0 else 0
        door_data.append([t, 2, door2_open])
    #-----------------------------------------------------------

    df_doors = pd.DataFrame(door_data, columns=['time', 'door_id', 'is_open'])
    df_doors.to_csv("../../Data/doors_status.csv", index=False)

    print("Saved Mock Agents Log and Door Status")
#=====================================================================

#=====================================================================
if __name__ == "__main__":
    generate_outbreak_data(timesteps=100)
#=====================================================================
