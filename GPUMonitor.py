import pynvml
import time
import matplotlib.pyplot as plt

class GPUMonitor:
    infos = []
    interval = 0.01
    duration = 10
    gpu_count = 0
    def __init__(self, interval=0.01, duration=10):
        self.initialize_pynvml()
        GPUMonitor.gpu_count = self.get_gpu_count()
        GPUMonitor.interval = interval
        GPUMonitor.duration = duration
        GPUMonitor.infos = [[] for _ in range(GPUMonitor.gpu_count)]

    def initialize_pynvml(self):
        """Initialize the pynvml library."""
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            print(f"Failed to initialize pynvml: {e}")
            exit(1)

    def get_gpu_count(self):
        """Return the number of GPUs available."""
        return pynvml.nvmlDeviceGetCount()

    @staticmethod
    def get_gpu_info(handle):
        """Get utilization and memory info for a GPU handle."""
        # Get utilization rates (GPU compute usage in %)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = util.gpu  # Compute utilization

        # Get memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used = mem_info.used / 1024**2  # Convert to MiB
        mem_total = mem_info.total / 1024**2  # Convert to MiB
        mem_percent = (mem_used / mem_total) * 100

        return gpu_util, mem_used, mem_total, mem_percent

    @staticmethod
    def monitor_gpus():
        """Monitor GPUs at specified intervals for a given duration."""
        gpu_count = GPUMonitor.gpu_count
        interval = GPUMonitor.interval
        duration = GPUMonitor.duration
        
        init_ts = time.time()
        for _ in range(int(duration / interval)):
            ts = time.time()
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).encode('utf-8')
                gpu_util, mem_used, mem_total, mem_percent = GPUMonitor.get_gpu_info(handle)
                
                GPUMonitor.infos[i].append({"timestamp": ts - init_ts, "gpu_util": gpu_util,
                                "mem_used": mem_used, "mem_total": mem_total, "mem_percent": mem_percent})
            
            time.sleep(interval)
        
        pynvml.nvmlShutdown()

    @staticmethod
    def plot_gpu_stats():
        """Plot GPU utilization and memory usage with y-axis from 0 to 100% and x-axis from 0 to max timestamp."""
        for i, info in enumerate(GPUMonitor.infos):
            # Extract data
            timestamps = [entry["timestamp"] for entry in info]
            gpu_utils = [entry["gpu_util"] for entry in info]
            mem_usages = [entry["mem_percent"] for entry in info]

            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot GPU Utilization
            plt.subplot(2, 1, 1)
            plt.plot(timestamps, gpu_utils, label='GPU Utilization (%)')
            plt.title(f'GPU {i} Utilization')
            plt.xlabel('Time (s)')
            plt.ylabel('Utilization (%)')
            plt.ylim(0, 100)  # Set y-axis from 0 to 100%
            plt.xlim(0, max(timestamps))  # Set x-axis from 0 to max timestamp
            plt.grid(True)
            plt.legend()

            # Plot Memory Usage
            plt.subplot(2, 1, 2)
            plt.plot(timestamps, mem_usages, label='Memory Usage (%)', color='orange')
            plt.title(f'GPU {i} Memory Usage')
            plt.xlabel('Time (s)')
            plt.ylabel('Memory Usage (%)')
            plt.ylim(0, 100)  # Set y-axis from 0 to 100%
            plt.xlim(0, max(timestamps))  # Set x-axis from 0 to max timestamp1
            print(max(timestamps))
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            # Save the figure
            plt.savefig(f'gpu_{i}_stats.png')
            plt.close()  # Close the figure to free memory
