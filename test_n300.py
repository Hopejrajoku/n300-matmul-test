import torch
import ttnn
import time
import os

def run_n300_benchmark():
    # Set the architecture programmatically just in case
    os.environ["ARCH_NAME"] = "wormhole_b0"
    
    device_id = 0
    device = None

    print(f"--- Tenstorrent N300 MatMul Benchmark ---")
    
    try:
        # 1. Initialize Device
        print(f"Initializing Device {device_id}...")
        device = ttnn.open_device(device_id=device_id)
        
        # 2. Prepare Input Tensors (1024x1024)
        print("Preparing 1024x1024 Tensors...")
        torch_a = torch.ones((1024, 1024), dtype=torch.bfloat16)
        torch_b = torch.ones((1024, 1024), dtype=torch.bfloat16)

        # 3. Move Tensors to Device
        tt_a = ttnn.from_torch(torch_a, device=device, layout=ttnn.TILE_LAYOUT)
        tt_b = ttnn.from_torch(torch_b, device=device, layout=ttnn.TILE_LAYOUT)

        # 4. Run MatMul (Warm-up)
        # We run it once to compile the kernels
        print("Compiling Kernels (Cold Start)...")
        ttnn.matmul(tt_a, tt_b)
        ttnn.synchronize(device)

        # 5. Profile MatMul (Timed Run)
        print("Running Timed Benchmark...")
        start_time = time.time()
        
        output_tt = ttnn.matmul(tt_a, tt_b)
        ttnn.synchronize(device)
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        print(f"\n Success!")
        print(f"Hardware Target: N300 (Wormhole B0)")
        print(f"Matrix Size: 1024 x 1024")
        print(f"Execution Time: {duration_ms:.2f} ms")

    except Exception as e:
        print(f"\n Execution Failed!")
        print(f"Error Details: {e}")
        print("\nTroubleshooting Tip: Ensure ttnn is installed (pip install ttnn) and")
        print("check hardware visibility with 'tt-smi'.")

    finally:
        if device:
            ttnn.close_device(device)
            print("\nDevice closed.")

if __name__ == "__main__":
    run_n300_benchmark()
