# Tenstorrent N300 MatMul Benchmark

This repository contains a high-performance matrix multiplication (MatMul) benchmark tailored for the Tenstorrent N300 (Wormhole B0) accelerator.

The script utilizes the ttnn library to execute a 1024x1024 bfloat16 operation directly on the device's Tensix cores.

# Quick Start
1. Environment Setup
Ensure you are using the correct Python environment and have the necessary libraries installed.

Bash

# Activate your Tenstorrent virtual environment
source /opt/venv/bin/activate

# Install the ttnn library
pip install ttnn
# 2. Hardware Configuration
The script targets the Wormhole B0 architecture. Ensure your environment variable is set:

Bash

export ARCH_NAME=wormhole_b0

# Run the Benchmark
Bash

python3 test_n300.py

# Benchmark Logic
To provide the most accurate hardware latency figures, the script implements the following logic:

Host-to-Device Transfer: Moves tensors from CPU memory to N300 SRAM.

Cold Start (Compilation): Executes a "warm-up" pass to compile the MatMul kernels.

Timed Execution: Uses time.perf_counter() and ttnn.synchronize() to measure pure hardware execution time without compilation overhead.

# Troubleshooting
If the hardware is not detected (TT_FATAL), verify the PCIe link using the Tenstorrent System Management Interface:

Bash

#List available devices
tt-smi -ls

#Reset the device if it is unresponsive
tt-smi -r 0
