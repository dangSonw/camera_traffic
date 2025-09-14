import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import logging
import sys
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LogAnalyzer:
    """Class to analyze and visualize YOLO performance logs"""
    
    def __init__(self, log_dir: str = "."):
        self.log_dir = Path(log_dir)
        self.data = {}
        self.config = {}
        
    def read_init_log(self) -> bool:
        """Read initialization configuration"""
        init_file = self.log_dir / "init_log.txt"
        if not init_file.exists():
            logger.warning(f"Init log file not found: {init_file}")
            return False
            
        try:
            with open(init_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line and not line.startswith('==='):
                        key, value = line.split(':', 1)
                        self.config[key.strip()] = value.strip()
            
            logger.info(f"Read configuration: {len(self.config)} parameters")
            return True
        except Exception as e:
            logger.error(f"Failed to read init log: {e}")
            return False
    
    def read_performance_logs(self) -> bool:
        """Read all performance log files"""
        log_files = {
            'time': 'time_log.txt',
            'fps': 'fps_log.txt',
            'cpu': 'system_cpu_log.txt',
            'ram': 'ram_log.txt',
        }
        
        # Success means: at least one file loaded successfully
        success = False
        for key, filename in log_files.items():
            file_path = self.log_dir / filename
            if not file_path.exists():
                logger.warning(f"Log file not found: {file_path}")
                continue
                
            try:
                with open(file_path, 'r') as f:
                    values = []
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                value = float(line)
                                values.append(value)
                            except ValueError:
                                logger.warning(f"Invalid value in {filename} line {line_num}: {line}")
                    
                    self.data[key] = np.array(values)
                    logger.info(f"Read {key} log: {len(values)} data points")
                    success = True
                    
            except Exception as e:
                logger.error(f"Failed to read {filename}: {e}")
                # keep going for other files
        
        return success and len(self.data) > 0
    
    def generate_time_axis(self) -> np.ndarray:
        """Generate X axis as frame indices (each point corresponds to one processed frame)."""
        max_len = max(len(v) for v in self.data.values()) if self.data else 0
        return np.arange(max_len)
    
    def calculate_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive statistics for all metrics"""
        stats = {}
        
        for key, values in self.data.items():
            if len(values) > 0:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                    'count': len(values)
                }
        
        return stats
    
    def create_summary_plots(self, save_path: Optional[str] = None, show_plot: bool = True):
        """Create comprehensive summary plots"""
        if not self.data:
            logger.error("No data available for plotting")
            return
        
        # Calculate time axis
        time_axis = self.generate_time_axis()
        max_points = len(time_axis)

        # Ensure all data arrays have the same length (pad/crop)
        for key in self.data:
            if len(self.data[key]) > max_points:
                self.data[key] = self.data[key][:max_points]
            elif len(self.data[key]) < max_points:
                last_val = self.data[key][-1] if len(self.data[key]) > 0 else 0
                self.data[key] = np.pad(
                    self.data[key],
                    (0, max_points - len(self.data[key])),
                    'constant', constant_values=last_val
                )

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('YOLO Performance Analysis Dashboard', fontsize=16, fontweight='bold')

        # Plot 1: Processing Time (only primary line)
        if 'time' in self.data and len(self.data['time']) > 0:
            axes[0, 0].plot(time_axis, self.data['time'], 'b-', linewidth=1)
            axes[0, 0].set_title('Frame Processing Time', fontweight='bold')
            axes[0, 0].set_xlabel('Frame')
            axes[0, 0].set_ylabel('Processing Time (ms)')
            axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: FPS (only primary line)
        if 'fps' in self.data and len(self.data['fps']) > 0:
            axes[0, 1].plot(time_axis, self.data['fps'], 'g-', linewidth=1)
            axes[0, 1].set_title('Frames Per Second', fontweight='bold')
            axes[0, 1].set_xlabel('Frame')
            axes[0, 1].set_ylabel('FPS')
            axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: CPU Usage (only primary line, filter zeros)
        if 'cpu' in self.data and len(self.data['cpu']) > 0:
            mask = self.data['cpu'] > 0
            cpu_data = self.data['cpu'][mask] if np.any(mask) else self.data['cpu']
            cpu_time = time_axis[mask] if np.any(mask) else time_axis
            if len(cpu_data) > 0:
                axes[1, 0].plot(cpu_time, cpu_data, color='orange', linewidth=1)
                axes[1, 0].set_title('CPU Usage', fontweight='bold')
                axes[1, 0].set_xlabel('Frame')
                axes[1, 0].set_ylabel('CPU Usage (%)')
                axes[1, 0].set_ylim(0, 100)
                axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: RAM Usage (only primary line)
        if 'ram' in self.data and len(self.data['ram']) > 0:
            axes[1, 1].plot(time_axis, self.data['ram'], color='purple', linewidth=1)
            axes[1, 1].set_title('RAM Usage', fontweight='bold')
            axes[1, 1].set_xlabel('Frame')
            axes[1, 1].set_ylabel('RAM Usage (MB)')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Summary plots saved to: {save_path}")

        if show_plot:
            plt.show()

        return fig
    
    def create_distribution_plots(self, save_path: Optional[str] = None, show_plot: bool = True):
        """Create distribution analysis plots"""
        if not self.data:
            logger.error("No data available for plotting")
            return
        
        metrics_with_data = [key for key in self.data if len(self.data[key]) > 0]
        if not metrics_with_data:
            logger.error("No valid data for distribution plots")
            return
        
        n_metrics = len(metrics_with_data)
        cols = 2
        rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
        fig.suptitle('Performance Metrics Distribution Analysis', fontsize=16, fontweight='bold')
        
        if rows == 1:
            axes = [axes] if n_metrics == 1 else axes
        else:
            axes = axes.flatten()
        
        colors = ['blue', 'green', 'orange', 'purple']
        
        for i, key in enumerate(metrics_with_data):
            data = self.data[key]
            
            if key == 'cpu':
                # Filter out zero values for CPU
                data = data[data > 0]
            
            if len(data) == 0:
                continue
                
            # Create histogram with KDE
            axes[i].hist(data, bins=50, density=True, alpha=0.7, 
                        color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
            
            # Add KDE curve
            try:
                from scipy import stats
                kde = stats.gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 200)
                axes[i].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            except ImportError:
                logger.warning("scipy not available, skipping KDE curve")
            
            # Add statistics
            mean_val = np.mean(data)
            std_val = np.std(data)
            median_val = np.median(data)
            
            axes[i].axvline(mean_val, color='red', linestyle='--', 
                           label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(median_val, color='green', linestyle='--', 
                           label=f'Median: {median_val:.2f}')
            
            # Set labels and title
            unit_map = {'time': 'ms', 'fps': 'FPS', 'cpu': '%', 'ram': 'MB'}
            unit = unit_map.get(key, '')
            
            axes[i].set_title(f'{key.upper()} Distribution', fontweight='bold')
            axes[i].set_xlabel(f'{key.capitalize()} ({unit})')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution plots saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def print_statistics_report(self):
        """Print comprehensive statistics report"""
        if not self.data:
            logger.error("No data available for statistics")
            return
        
        stats = self.calculate_statistics()
        
        print("\n" + "="*60)
        print("📊 YOLO PERFORMANCE STATISTICS REPORT")
        print("="*60)
        
        # Configuration info
        if self.config:
            print("\n🔧 Configuration:")
            key_configs = ['Source', 'Weights', 'Target FPS', 'Used cores', 'Model stride']
            for key in key_configs:
                if key in self.config:
                    print(f"  {key}: {self.config[key]}")
        
        # Performance statistics
        print("\n Performance Metrics:")
        
        for metric, stat_dict in stats.items():
            unit_map = {'time': 'ms', 'fps': 'FPS', 'cpu': '%', 'ram': 'MB'}
            unit = unit_map.get(metric, '')
            
            print(f"\n  {metric.upper()} ({unit}):")
            print(f"    Mean: {stat_dict['mean']:.2f} ± {stat_dict['std']:.2f}")
            print(f"    Range: {stat_dict['min']:.2f} - {stat_dict['max']:.2f}")
            print(f"    Median: {stat_dict['median']:.2f}")
            print(f"    95th percentile: {stat_dict['p95']:.2f}")
            print(f"    99th percentile: {stat_dict['p99']:.2f}")
            print(f"    Data points: {stat_dict['count']:,}")
        
        print("\n" + "="*60)

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Analyze and visualize YOLO performance logs")
    parser.add_argument('--log-dir', type=str, default='.', 
                       help='Directory containing log files')
    parser.add_argument('--save-summary', type=str, default=None,
                       help='Save summary plots to file (e.g., summary.png)')
    parser.add_argument('--save-distribution', type=str, default=None,
                       help='Save distribution plots to file (e.g., distribution.png)')
    parser.add_argument('--no-show', action='store_true',
                       help='Don\'t show plots (useful for batch processing)')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only print statistics, no plots')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = LogAnalyzer(args.log_dir)
    
    # Read configuration
    config_read = analyzer.read_init_log()
    if not config_read:
        logger.warning("Could not read configuration, continuing with data analysis...")
    
    # Read performance data
    if not analyzer.read_performance_logs():
        logger.error("Failed to read performance logs")
        return 1
    
    # Print statistics report
    analyzer.print_statistics_report()
    
    # Generate plots if requested
    if not args.stats_only:
        show_plots = not args.no_show
        
        # Summary plots: show interactively if allowed
        try:
            analyzer.create_summary_plots(args.save_summary, show_plots)
        except Exception as e:
            logger.error(f"Failed to create summary plots: {e}")
        
        # Distribution plots: avoid opening a second window; only save if requested
        try:
            if args.save_distribution or args.no_show:
                analyzer.create_distribution_plots(args.save_distribution, show_plot=False)
        except Exception as e:
            logger.error(f"Failed to create distribution plots: {e}")
    
    logger.info("Analysis completed successfully")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)