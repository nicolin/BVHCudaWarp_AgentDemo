**NVIDIA Warp (CUDA) Physics Engine and Visualizer**

This repository contains a high-performance Agent-Based Model (ABM) designed to simulate human movement and disease transmission within a hospital environment. By offloading spatial queries to the GPU using **NVIDIA Warp**, the engine can process thousands of agents, execute BVH (Bounding Volume Hierarchy) wall-collisions, and calculate HashGrid proximity transmissions at scale.

Included in this repository is the **Sim Viewer**, a lightweight, web-based dashboard built with Plotly and Dash to visualize the simulation's output, complete with comet-trail tracking and a headless MP4 video exporter.


## ✨ Key Features

* **GPU-Accelerated Physics:** Uses CUDA (via NVIDIA Warp) to handle kinematic agent movement, BVH ray-cast collision detection against CAD floorplans, and HashGrid spatial queries for disease spread.
* **Semantic CAD Integration:** Directly ingests compiled `.npz` hospital floorplans, automatically extracting solid walls, interactive doors, room labels, and hospital beds.
* **Dashboard:** A browser-based interactive UI featuring:
  * Smooth playback and scrubbable timeline.
  * "Comet-tail" historical pathing (downsampled for massive performance).
  * An Event Locator dropdown to instantly snap the camera to transmission events.
* **Headless MP4 Export:** A built-in OpenCV rendering pipeline to silently stitch simulation data into high-quality 24fps `.mp4` video files.



## ⚙️ Installation & Setup

### Prerequisites
* **Python 3.8+**
* An NVIDIA GPU with updated drivers (Required for Warp CUDA execution).

### 1. Clone the Repository

git clone [https://github.com/nicolin/BVHCudaWarp_AgentDemo.git](https://github.com/nicolin/BVHCudaWarp_AgentDemo.git)
cd BVHCudaWarp_AgentDemo


### 2. Install Dependencies
Install the required Python packages for the GPU engine and the web visualizer:
pip install warp-lang pandas numpy dash plotly kaleido opencv-python Pillow tqdm
*(Note: `kaleido`, `opencv-python`, and `Pillow` are specifically required for the MP4 video export feature).*

##  Mock Data Generation
### 1. CAD
* Mock_DXFGen.py: creates a DXF file, you can modify this if you do not have any floor plans.
* convertor_dxf_npz.py: processes it into a npz BVH file that is read by run_sim.py 
* Util_CADView.py: is a lightweight viewer to check the npz file and launch a test query. 

### 2. Agent Logic
Mock_AgentLog.py creates a mock csv of agents motions along with door status 


## 🚀 How to Run
### 1. Run the GPU Simulation
Execute the main physics engine to calculate agent trajectories and transmissions.
python3 src/Simulation/run_sim.py

**Outputs Generated:**
* simulation_outputs/processed_telemetry.csv (Agent coordinates per tick)
* simulation_outputs/transmission_events.csv (Transmission events)

### 2. Launch the Visualizer
Once the simulation is complete, launch the interactive Digital Twin viewer:

python src/simulation/Util_SimDash.py

1. Open your web browser and navigate to **`http://HostName:8080`**.
2. **Play/Pause:** Use the UI controls to watch the agents navigate the hospital.
3. **Jump to Event:** Use the dropdown menu to instantly snap the timeline to an infection transmission.
4. **Export Video:** Click the red **🎬 Export to MP4** button to render the simulation into a video file. Look at your terminal for the render progress bar!

## 👨‍💻 Author
**Nicolin Govender**
