# Adaptive Variable Resolution 2.5D LiDAR Mapping

## Description

Adaptive Variable Resolution 2.5D LiDAR Mapping is a perception pipeline for dynamic environments, designed around an edge-deployed unmanned ground vehicle (UGV). The system transforms raw LiDAR point clouds into a variable-resolution 2.5D elevation map with semantic layers.

The proposed architecture combines:

- **Spiking PointNet++** for semantic segmentation of drivable terrain, static obstacles, and dynamic objects.
- **Variable-resolution 2.5D mapping** with 5 cm cells within 10 m and 50 cm cells from 10 m to 100 m.
- **Quadtree-based spatial representation** to maintain alignment between resolutions and avoid projection gaps or overlaps.
- **Event-driven grid updates** so cells are updated when relevant spike events occur.
- **Streamlit visualization** for a real-time top-down tactical map and performance metrics.
- **Performance profiling** comparing conventional Multiply-Accumulate (MAC) operations with SNN Accumulate-only (AC) operations.

The design uses PyTorch and a neuromorphic library such as snnTorch, with NumPy/Open3D-based point-cloud processing and a Streamlit dashboard.

## Installation

The source documentation defines the following implementation stack but does not provide a complete, pinned dependency list or a single installation command.

The project requires a Python environment capable of running the proposed stack, including:

- PyTorch
- snnTorch
- NumPy
- Open3D
- Streamlit
- Plotly and/or PyVista
- SciPy
- Numba
- `torchprofile` and/or `fvcore`
- `psutil`
- Matplotlib

Additional point-cloud utilities mentioned in the documentation include `laspy` and `pypcd`. `torch-geometric` is also identified as a possible component of the model stack.

Because exact package versions and a finalized dependency file are not specified in the source documentation, the environment should be pinned when the implementation is finalized.

## Usage

The intended processing pipeline is:

1. **LiDAR ingestion and preprocessing**
   - Load and clean LiDAR point clouds.
   - Convert point coordinates and intensity into the model's temporal representation.
   - Apply sampling and spatial grouping as required.

2. **Spiking PointNet++ inference**
   - Process point-cloud data using PointNet++-style hierarchical spatial grouping.
   - Replace conventional ReLU activations with Leaky Integrate-and-Fire (LIF) neurons.
   - Produce semantic predictions for:
     - Drivable terrain
     - Static obstacles
     - Dynamic objects
   - Expose spike information for downstream event-driven processing.

3. **Variable-resolution 2.5D grid generation**
   - Use 5 cm cells for points within 10 m of the sensor.
   - Use 50 cm cells for points from 10 m to 100 m.
   - Represent the hierarchy with a Quadtree so the coarse cells are exact parents of the fine cells.
   - Store elevation and semantic class information for grid cells.
   - Update cells based on non-zero SNN spike events.

4. **Visualization**
   - Run the Streamlit dashboard.
   - Display the environment from a top-down perspective.
   - Represent drivable terrain, static obstacles, and dynamic objects using the semantic color scheme defined by the project.
   - Display variable grid resolution and live performance indicators.

5. **Performance evaluation**
   - Profile MAC and AC operation counts.
   - Measure latency, FPS, and memory usage.
   - Compare the baseline PointNet++ implementation with Spiking PointNet++.
   - Evaluate behavior under simulated edge-device resource constraints.

The documentation does not specify a final command for launching the application because the repository structure and entry-point filenames have not yet been defined.

## Contributing

Contributions should be organized around the modular project architecture. The proposed implementation separates responsibilities into:

- **Neuromorphic Deep Learning:** Spiking PointNet++ architecture and training.
- **Point-Cloud and Temporal Data:** LiDAR ingestion, preprocessing, temporal encoding, and dataset preparation.
- **Spatial Algorithms:** Variable-resolution grid generation and Quadtree implementation.
- **Visualization:** Streamlit dashboard and real-time map rendering.
- **Profiling and Edge Benchmarking:** MAC/AC profiling, latency, FPS, memory, and resource-constrained evaluation.
- **Systems Integration:** End-to-end integration, testing, repository workflow, and demonstration preparation.

During implementation, contributors should agree on interface formats and tensor/array shapes before integrating modules. The documented development plan uses three phases:

1. **Mock and Build:** Establish interfaces and develop components independently using dummy data.
2. **Integration:** Connect the data pipeline, SNN model, grid engine, and dashboard.
3. **Benchmarking and Presentation:** Run edge-constrained benchmarks, finalize performance measurements, and prepare the demonstration.

## Guide

1.**Clone the project**

```bash
git clone https://github.com/ProgrammerAdi-369/AVRLM.git
```

2. **Enter project**

```bash
cd AVRLM
```

3.**Get latest changes**

```bash
git pull origin main
```

4.**Work on the project**
Save your changes to Git

```bash
git add .
git commit -m "Describe your changes"
git push origin main
```

## License

MIT License
Copyright (c) 2026 AVRLM Contributors
