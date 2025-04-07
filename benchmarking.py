import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns
import torch
import math
# Import the original einops and your custom implementation
from einops import rearrange as einops_rearrange
# Import your implementation - make sure this path is correct
from rearrange import rearrange as custom_rearrange

def benchmark_operation(implementation, name, operation, array, pattern, is_torch_implementation=True, **kwargs):
    """Benchmark a single rearrange operation."""
    # Check if we need to convert between PyTorch and NumPy
    if not is_torch_implementation and isinstance(array, torch.Tensor):
        # Convert PyTorch tensor to NumPy for custom implementation
        numpy_array = array.numpy()
        
        # Warmup
        for _ in range(3):
            implementation(numpy_array.copy(), pattern, **kwargs)
        
        # Actual timing
        times = []
        iterations = 10
        for _ in range(iterations):
            start = time.perf_counter()
            implementation(numpy_array.copy(), pattern, **kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    else:
        # Use tensor directly for torch implementations
        # Warmup
        for _ in range(3):
            implementation(array.clone(), pattern, **kwargs)
        
        # Actual timing
        times = []
        iterations = 10
        for _ in range(iterations):
            start = time.perf_counter()
            implementation(array.clone(), pattern, **kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'implementation': name,
        'operation': operation,
        'shape': str(array.shape),
        'pattern': pattern,
        'mean_time_ms': np.mean(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'std_time_ms': np.std(times)
    }

def run_benchmarks(repeat=10):
    """Run all benchmarks and return results as a list of dictionaries."""
    all_results = []
    
    benchmark_cases = [
        {
            'name': 'Simple reshape',
            'pattern': 'b c -> c b',
            'shapes': [(10, 20), (100, 200), (1000, 2000)],
            'kwargs': [{}, {}, {}]  # Same empty kwargs for all shapes
        },
        {
            'name': 'Split dimension',
            'pattern': '(a b) c -> a b c',
            'shapes': [(6, 4), (60, 40), (600, 400)],
            'kwargs': [
                {'a': 2, 'b': 3},     # For shape (6, 4)
                {'a': 20, 'b': 3},    # For shape (60, 40)
                {'a': 200, 'b': 3}    # For shape (600, 400)
            ]
        },
        {
            'name': 'Merge dimensions',
            'pattern': 'a b c -> (a b) c',
            'shapes': [(2, 3, 4), (20, 30, 40), (200, 30, 40)],
            'kwargs': [{}, {}, {}]  # Same empty kwargs for all shapes
        },
        {
            'name': 'Wildcard handling',
            'pattern': 'a b c -> b c a',  # Changed from '*' to named dimension
            'shapes': [(2, 3, 4), (20, 30, 40), (200, 30, 40)],
            'kwargs': [{}, {}, {}]  # Same empty kwargs for all shapes
        },
        {
            'name': 'Complex pattern',
            'pattern': 'b (c d) e -> (b e) d c',
            'shapes': [(2, 6, 3), (20, 60, 30), (200, 60, 30)],
            'kwargs': [
                {'c': 2, 'd': 3},     # For shape (2, 6, 3)
                {'c': 20, 'd': 3},    # For shape (20, 60, 30)
                {'c': 20, 'd': 3}     # For shape (200, 60, 30)
            ]
        }
    ]
    
    # Run benchmarks
    for case in benchmark_cases:
        for i, shape in enumerate(case['shapes']):
            # Get the kwargs for this specific shape
            kwargs = case['kwargs'][i] if isinstance(case['kwargs'], list) else case['kwargs']
            
            # Create random tensor with the specified shape
            tensor = torch.randn(*shape)
            
            # Make sure tensor is on CPU and contiguous for fair comparison
            tensor = tensor.cpu().contiguous()
            
            # Benchmark original einops (PyTorch implementation)
            try:
                result_original = benchmark_operation(
                    einops_rearrange, 
                    'Original einops', 
                    case['name'], 
                    tensor, 
                    case['pattern'],
                    is_torch_implementation=True,
                    **kwargs
                )
                all_results.append(result_original)
            except Exception as e:
                print(f"Error with original einops for {case['name']}, shape {shape}: ", e)
            
            # Benchmark custom implementation (NumPy implementation)
            try:
                result_custom = benchmark_operation(
                    custom_rearrange, 
                    'Custom implementation', 
                    case['name'], 
                    tensor, 
                    case['pattern'],
                    is_torch_implementation=False,  # Indicate this uses NumPy
                    **kwargs
                )
                all_results.append(result_custom)
            except Exception as e:
                print(f"Error with custom implementation for {case['name']}, shape {shape}: ", e)
    
    return pd.DataFrame(all_results)

def generate_report(results_df):
    """Generate a performance comparison report."""
    # Calculate speedup/slowdown
    pivot_df = results_df.pivot_table(
        index=['operation', 'shape'],
        columns='implementation',
        values='mean_time_ms'
    )
    
    # Check if both implementations exist in the results
    if 'Original einops' in pivot_df.columns and 'Custom implementation' in pivot_df.columns:
        pivot_df['speedup'] = pivot_df['Original einops'] / pivot_df['Custom implementation']
        pivot_df['relative_diff_percent'] = (pivot_df['Custom implementation'] / pivot_df['Original einops'] - 1) * 100
    else:
        # Handle case where one implementation might be missing
        print("Warning: One of the implementations might be missing from results")
        if 'Original einops' not in pivot_df.columns:
            pivot_df['Original einops'] = float('nan')
        if 'Custom implementation' not in pivot_df.columns:
            pivot_df['Custom implementation'] = float('nan')
        pivot_df['speedup'] = float('nan')
        pivot_df['relative_diff_percent'] = float('nan')
    
    # Generate report text
    report = "# Performance Comparison: Custom einops vs. Original einops\n\n"
    
    # Summary statistics
    avg_speedup = pivot_df['speedup'].mean()
    if not math.isnan(avg_speedup):
        if avg_speedup > 1:
            report += f"## Summary\nOn average, the custom implementation is **{avg_speedup:.2f}x faster** than the original einops library.\n\n"
        else:
            report += f"## Summary\nOn average, the custom implementation is **{1/avg_speedup:.2f}x slower** than the original einops library.\n\n"
    else:
        report += "## Summary\nUnable to calculate average speedup due to missing data.\n\n"
    
    # Detailed results table
    report += "## Detailed Performance Results\n\n"
    table = pivot_df.reset_index()
    report += tabulate(table, headers='keys', tablefmt='pipe') + "\n\n"
    
    # Generate plots
    plt.figure(figsize=(12, 8))
    
    # Bar chart comparison
    comparison_df = results_df.pivot_table(
        index=['operation', 'shape'],
        columns='implementation',
        values='mean_time_ms'
    ).reset_index()
    
    # Reshape for seaborn
    melted_df = pd.melt(
        comparison_df, 
        id_vars=['operation', 'shape'], 
        value_vars=[col for col in comparison_df.columns if col not in ['operation', 'shape']],
        var_name='Implementation', value_name='Time (ms)'
    )
    
    plt.figure(figsize=(15, 10))
    sns.barplot(x='shape', y='Time (ms)', hue='Implementation', data=melted_df)
    plt.title('Performance Comparison: Custom vs Original einops')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    
    # Add chart reference to report
    report += "## Visual Comparison\n\n"
    report += "![Performance Comparison](performance_comparison.png)\n\n"
    
    # Operation-specific analysis
    report += "## Operation-Specific Analysis\n\n"
    for operation in results_df['operation'].unique():
        op_data = results_df[results_df['operation'] == operation]
        custom_data = op_data[op_data['implementation'] == 'Custom implementation']
        original_data = op_data[op_data['implementation'] == 'Original einops']
        
        # Check if both implementations have data
        if not custom_data.empty and not original_data.empty:
            # Calculate average speedup for this operation
            avg_op_speedup = original_data['mean_time_ms'].mean() / custom_data['mean_time_ms'].mean()
            
            report += f"### {operation}\n"
            if avg_op_speedup > 1:
                report += f"For {operation}, the custom implementation is on average **{avg_op_speedup:.2f}x faster**.\n\n"
            else:
                report += f"For {operation}, the custom implementation is on average **{1/avg_op_speedup:.2f}x slower**.\n\n"
        else:
            report += f"### {operation}\n"
            report += "Insufficient data to compare implementations for this operation.\n\n"
    
    # Scaling analysis
    report += "## Scaling Analysis\n\n"
    report += "This section analyzes how performance differences scale with input size.\n\n"
    
    for operation in results_df['operation'].unique():
        op_data = results_df[results_df['operation'] == operation]
        shapes = op_data['shape'].unique()
        
        if len(shapes) > 1:  # Only include if we have multiple shapes
            report += f"### {operation}\n"
            
            # Check if this operation exists in the pivot table
            if operation in pivot_df.index.get_level_values(0):
                op_pivot = pivot_df.loc[operation].reset_index()
                report += tabulate(op_pivot, headers='keys', tablefmt='pipe') + "\n\n"
                
                # Check if we have enough non-NaN values to calculate trend
                if op_pivot['relative_diff_percent'].notna().sum() > 1:
                    trend = op_pivot['relative_diff_percent'].diff().mean()
                    if trend > 0:
                        report += f"As input size increases, the custom implementation gets **relatively slower** ({trend:.2f}% per step).\n\n"
                    elif trend < 0:
                        report += f"As input size increases, the custom implementation gets **relatively faster** ({-trend:.2f}% per step).\n\n"
                    else:
                        report += "The relative performance remains consistent across different input sizes.\n\n"
                else:
                    report += "Insufficient data to analyze performance scaling.\n\n"
            else:
                report += "Insufficient data to analyze performance scaling.\n\n"
    
    # Conclusion
    report += "## Conclusion\n\n"
    if not math.isnan(avg_speedup):
        if avg_speedup > 1.1:
            report += "The custom implementation shows significant performance improvements over the original einops library, particularly for certain operations. "
        elif avg_speedup < 0.9:
            report += "The original einops library generally outperforms the custom implementation. This suggests there may be optimization opportunities in the custom code. "
        else:
            report += "The custom implementation performs similarly to the original einops library. This is a solid achievement for a from-scratch implementation. "
    else:
        report += "Insufficient data to draw overall conclusions about performance. "
    
    # Recommendations based on results
    report += "\n\n### Recommendations\n\n"
    try:
        slowest_ops = pivot_df[pivot_df['speedup'] < 0.9].index.get_level_values(0).unique()
        if len(slowest_ops) > 0:
            report += "Consider optimizing these operations in the custom implementation:\n"
            for op in slowest_ops:
                report += f"- {op}\n"
        else:
            report += "No specific operations identified for optimization priority.\n"
    except Exception:
        report += "Unable to identify specific operations for optimization.\n"
    
    return report

if __name__ == "__main__":
    print("Starting benchmark...")
    results = run_benchmarks()
    print("Generating report...")
    report = generate_report(results)
    
    # Save results and report
    results.to_csv('benchmark_results.csv', index=False)
    with open('performance_report.md', 'w') as f:
        f.write(report)
    
    print("Benchmarking complete! Results saved to 'benchmark_results.csv'")
    print("Performance report generated as 'performance_report.md'")
    
    # Print summary to console
    try:
        pivot_df = results.pivot_table(
            index=['operation', 'shape'],
            columns='implementation',
            values='mean_time_ms'
        )
        
        if 'Original einops' in pivot_df.columns and 'Custom implementation' in pivot_df.columns:
            pivot_df['speedup'] = pivot_df['Original einops'] / pivot_df['Custom implementation']
            avg_speedup = pivot_df['speedup'].mean()
            
            if not math.isnan(avg_speedup):
                if avg_speedup > 1:
                    print(f"\nSummary: Custom implementation is {avg_speedup:.2f}x faster on average")
                else:
                    print(f"\nSummary: Custom implementation is {1/avg_speedup:.2f}x slower on average")
            else:
                print("\nSummary: Could not calculate average speedup due to missing or invalid data")
        else:
            print("\nSummary: Could not calculate speedup - one or both implementations are missing from results")
    except Exception as e:
        print(f"\nError calculating summary statistics: {e}")