# Performance Comparison: Custom einops vs. Original einops

## Summary

On average, the custom implementation is **2.42x slower** than the original einops library.

## Detailed Performance Results

|     | operation         | shape                     | Custom implementation | Original einops |   speedup | relative_diff_percent |
| --: | :---------------- | :------------------------ | --------------------: | --------------: | --------: | --------------------: |
|   0 | Complex pattern   | torch.Size([2, 6, 3])     |               0.01365 |         0.01113 |  0.815385 |               22.6415 |
|   1 | Complex pattern   | torch.Size([20, 60, 30])  |               0.03096 |         0.02525 |  0.815569 |               22.6138 |
|   2 | Complex pattern   | torch.Size([200, 60, 30]) |               0.87604 |         0.07028 | 0.0802246 |                1146.5 |
|   3 | Merge dimensions  | torch.Size([2, 3, 4])     |               0.01066 |         0.00415 |  0.389305 |               156.868 |
|   4 | Merge dimensions  | torch.Size([20, 30, 40])  |               0.01261 |          0.0055 |  0.436162 |               129.273 |
|   5 | Merge dimensions  | torch.Size([200, 30, 40]) |               0.03201 |         0.00903 |  0.282099 |               254.485 |
|   6 | Simple reshape    | torch.Size([10, 20])      |               0.00963 |          0.0048 |  0.498442 |               100.625 |
|   7 | Simple reshape    | torch.Size([100, 200])    |               0.01129 |         0.00512 |  0.453499 |               120.508 |
|   8 | Simple reshape    | torch.Size([1000, 2000])  |                1.8218 |         0.12037 |  0.066072 |                1413.5 |
|   9 | Split dimension   | torch.Size([6, 4])        |               0.01139 |         0.00536 |  0.470589 |                 112.5 |
|  10 | Split dimension   | torch.Size([60, 40])      |               0.01139 |      0.00515999 |  0.453028 |               120.737 |
|  11 | Split dimension   | torch.Size([600, 400])    |               0.03666 |         0.00946 |  0.258047 |               287.527 |
|  12 | Wildcard handling | torch.Size([2, 3, 4])     |               0.01033 |         0.00433 |  0.419167 |               138.568 |
|  13 | Wildcard handling | torch.Size([20, 30, 40])  |               0.01208 |         0.00562 |  0.465232 |               114.947 |
|  14 | Wildcard handling | torch.Size([200, 30, 40]) |               0.03335 |         0.00977 |  0.292954 |               241.351 |

## Visual Comparison

![Performance Comparison](performance_comparison.png)

## Operation-Specific Analysis

### Simple reshape

For Simple reshape, the custom implementation is on average **14.14x slower**.

### Split dimension

For Split dimension, the custom implementation is on average **2.97x slower**.

### Merge dimensions

For Merge dimensions, the custom implementation is on average **2.96x slower**.

### Wildcard handling

For Wildcard handling, the custom implementation is on average **2.83x slower**.

### Complex pattern

For Complex pattern, the custom implementation is on average **8.63x slower**.

## Scaling Analysis

This section analyzes how performance differences scale with input size.

### Simple reshape

|     | shape                    | Custom implementation | Original einops |  speedup | relative_diff_percent |
| --: | :----------------------- | --------------------: | --------------: | -------: | --------------------: |
|   0 | torch.Size([10, 20])     |               0.00963 |          0.0048 | 0.498442 |               100.625 |
|   1 | torch.Size([100, 200])   |               0.01129 |         0.00512 | 0.453499 |               120.508 |
|   2 | torch.Size([1000, 2000]) |                1.8218 |         0.12037 | 0.066072 |                1413.5 |

As input size increases, the custom implementation gets **relatively slower** (656.44% per step).

### Split dimension

|     | shape                  | Custom implementation | Original einops |  speedup | relative_diff_percent |
| --: | :--------------------- | --------------------: | --------------: | -------: | --------------------: |
|   0 | torch.Size([6, 4])     |               0.01139 |         0.00536 | 0.470589 |                 112.5 |
|   1 | torch.Size([60, 40])   |               0.01139 |      0.00515999 | 0.453028 |               120.737 |
|   2 | torch.Size([600, 400]) |               0.03666 |         0.00946 | 0.258047 |               287.527 |

As input size increases, the custom implementation gets **relatively slower** (87.51% per step).

### Merge dimensions

|     | shape                     | Custom implementation | Original einops |  speedup | relative_diff_percent |
| --: | :------------------------ | --------------------: | --------------: | -------: | --------------------: |
|   0 | torch.Size([2, 3, 4])     |               0.01066 |         0.00415 | 0.389305 |               156.868 |
|   1 | torch.Size([20, 30, 40])  |               0.01261 |          0.0055 | 0.436162 |               129.273 |
|   2 | torch.Size([200, 30, 40]) |               0.03201 |         0.00903 | 0.282099 |               254.485 |

As input size increases, the custom implementation gets **relatively slower** (48.81% per step).

### Wildcard handling

|     | shape                     | Custom implementation | Original einops |  speedup | relative_diff_percent |
| --: | :------------------------ | --------------------: | --------------: | -------: | --------------------: |
|   0 | torch.Size([2, 3, 4])     |               0.01033 |         0.00433 | 0.419167 |               138.568 |
|   1 | torch.Size([20, 30, 40])  |               0.01208 |         0.00562 | 0.465232 |               114.947 |
|   2 | torch.Size([200, 30, 40]) |               0.03335 |         0.00977 | 0.292954 |               241.351 |

As input size increases, the custom implementation gets **relatively slower** (51.39% per step).

### Complex pattern

|     | shape                     | Custom implementation | Original einops |   speedup | relative_diff_percent |
| --: | :------------------------ | --------------------: | --------------: | --------: | --------------------: |
|   0 | torch.Size([2, 6, 3])     |               0.01365 |         0.01113 |  0.815385 |               22.6415 |
|   1 | torch.Size([20, 60, 30])  |               0.03096 |         0.02525 |  0.815569 |               22.6138 |
|   2 | torch.Size([200, 60, 30]) |               0.87604 |         0.07028 | 0.0802246 |                1146.5 |

As input size increases, the custom implementation gets **relatively slower** (561.93% per step).

## Conclusion

The original einops library generally outperforms the custom implementation. This suggests there may be optimization opportunities in the custom code.
