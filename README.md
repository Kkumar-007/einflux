# einflux

![einflux logo](https://img.shields.io/badge/einflux-reimagining%20einops-blue)
[![PyPI version](https://img.shields.io/badge/pypi-coming%20soon-orange)](https://github.com/Kkumar-007/einflux)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**einflux** is a Python library that provides a ground-up reimplementation of the core functionality of [einops](https://github.com/arogozhnikov/einops), offering tensor operations using the Einstein-inspired notation pattern specifically for NumPy arrays. This library aims to make NumPy array manipulations more intuitive, readable, and less error-prone.

## What is einflux?

einflux provides a clean, intuitive API for tensor manipulations with NumPy arrays. It uses a consistent notation pattern to express complex tensor operations that would otherwise require multiple steps with traditional NumPy operations like reshape and transpose.

```python
# Instead of this (NumPy):
batch_size = x.shape[0]
height, width = x.shape[1:3]
x = x.reshape(batch_size, height * width, -1)

# Write this with einflux:
x = rearrange(x, 'b h w c -> b (h w) c')
```

## Installation

```bash
pip install einflux
```

For development installation:

```bash
git clone https://github.com/Kkumar-007/einflux.git
cd einflux
pip install -e .
```

## Usage

Basic usage example:

```python
import numpy as np
from einflux import rearrange

# Create a sample 4D tensor (batch, height, width, channels)
x = np.random.randn(32, 64, 64, 3)

# Rearrange to (batch, channels, height, width) format
x_rearranged = rearrange(x, 'b h w c -> b c h w')

# Flatten spatial dimensions
x_flattened = rearrange(x, 'b h w c -> b (h w) c')

# Split channel dimension into multiple heads
x_multihead = rearrange(x, 'b h w (n c) -> b n h w c', n=3)
```

## Key Features

- **Einstein-inspired notation**: Express complex tensor operations in a single, readable line
- **NumPy compatibility**: Works with NumPy arrays for scientific computing and data analysis
- **Error checking**: Validates array shapes against patterns at runtime with helpful error messages
- **Flexible dimension handling**: Supports both named dimensions and grouping with parentheses
- **Pure Python implementation**: Built from scratch with minimal dependencies (only NumPy)

## Design Philosophy and Implementation Approach

einflux was built with several key design principles in mind:

### 1. Readability and Explicitness

The core philosophy behind einflux is to make tensor manipulations more readable and self-documenting. The implementation prioritizes clarity and explicitness over extreme optimization, allowing developers to understand and maintain code that uses these tensor operations.

### 2. Pattern-Based Parser Architecture

At the heart of einflux is a robust pattern parser that systematically processes Einstein-inspired notation:

- **Tokenization**: The parser first breaks down patterns like `'b h w c -> b (h w) c'` into tokens and structured groups
- **Dimension Tracking**: The implementation carefully tracks named dimensions across the input and output patterns
- **Shape Inference**: Where possible, the system infers dimensions automatically and validates shapes

### 3. Three-Step Transformation Process

The core `rearrange` function follows a three-step transformation approach:

1. **Split dimensions** according to the input pattern, breaking grouped dimensions into individual ones
2. **Permute dimensions** to match the order specified in the output pattern
3. **Merge dimensions** according to the output pattern's grouping structure

This approach provides a systematic way to handle complex reshaping operations in a single call.

### 4. Comprehensive Error Handling

einflux includes detailed error messages through a custom `RearrangeError` class. The implementation checks for various failure cases:

- Dimension count mismatches
- Shape inconsistencies
- Invalid dimension groupings
- Inference failures

Each error provides detailed context to help users quickly identify and fix issues.

### 5. Dynamic Dimension Inference

The implementation can infer dimension sizes in several scenarios:

- When splitting a dimension, it can infer one unknown sub-dimension
- When exactly one output dimension is unknown, it can be inferred from the total size

## Core Operation: `rearrange`

The main function provided by einflux is `rearrange`, which allows you to reshape and permute NumPy array dimensions in one intuitive operation:

```python
# Flatten image dimensions
rearrange(x, 'b h w c -> b (h w) c')

# Matrix transposition
rearrange(x, 'i j -> j i')

# Reshape with dimension inference
rearrange(x, 'b n (h d) -> b n h d', h=8)

# Using numeric values in patterns
rearrange(x, '(b 2) c h w -> b c 2 h w')
```

### Pattern Syntax

- **Named dimensions**: Use any name (e.g., `b`, `h`, `w`, `c`) to represent a dimension
- **Grouping with parentheses**: Group dimensions to be split or merged, e.g., `(h w)`
- **Numeric values**: Specify fixed sizes within patterns, e.g., `(b 3)`
- **Special dimension placeholders**: Use `_` for dimensions you want to ignore

## Examples

### Image Processing

```python
import numpy as np
from einflux import rearrange

# Reshape image batch from NHWC to NCHW format
images = np.zeros((32, 224, 224, 3))  # NHWC format
x = rearrange(images, 'b h w c -> b c h w')  # Convert to NCHW

# Extract image patches
patches = rearrange(images, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=8, p2=8)
```

### Data Processing

```python
# Reshape for batch processing
data = np.random.randn(64, 10)
batched = rearrange(data, '(b n) d -> b n d', b=8)

# Reshape time series data
timeseries = np.random.randn(1000, 5)
windows = rearrange(timeseries, '(b s) f -> b s f', s=100)  # Create windows of 100 steps
```

### Linear Algebra

```python
# Matrix reshaping
matrix = np.random.randn(10, 20)
reshaped = rearrange(matrix, 'i j -> (i j)')  # Flatten to vector
restored = rearrange(reshaped, '(i j) -> i j', i=10, j=20)  # Restore shape
```

## Running Tests

To run the test suite:

```bash
# Clone the repository if you haven't already
git clone https://github.com/Kkumar-007/einflux.git
cd einflux

# Install test dependencies
pip install pytest

# Run the tests
pytest tests/
```

## Performance Benchmarks

## Benchmark Report:
I ran performance comparisons between einflux and the original einops library across various tensor sizes and rearrangement patterns.
 
On average, einflux is ~2.42x slower than einops.
 
This performance difference is expected due to the additional parsing logic and lack of low-level optimization in the current prototype. Future versions aim to narrow this gap while maintaining readability and flexibility.
 
📝 The full performance report is included in performance_report.md

### Running Benchmarks

To run the benchmarks comparing einflux with einops:

```bash
# Install benchmark dependencies
pip install einops pytest-benchmark

# Run benchmarks
python benchmarks/run_benchmarks.py
```

## Advanced Usage

### Dimension Inference

einflux can infer one unknown dimension automatically:

```python
# Infer the number of heads 'h' given dimension size 'd'
x = np.random.randn(8, 16, 512)  # [batch, tokens, embedding_dim]
y = rearrange(x, 'b t (h d) -> b h t d', d=64)
# 'h' is inferred to be 8 (512 / 64 = 8)
```

### Using Named Dimensions With Known Sizes

```python
x = np.random.randn(30, 40, 50)
# Specify dimension sizes directly in kwargs
result = rearrange(x, 'a b c -> (a c) b', a=30, c=50)
```

## Python & NumPy Version Support

einflux is compatible with:
- Python 3.7 and above
- NumPy 1.19 and above

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Development

Planned features for future versions:
- Performance optimizations to narrow the gap with einops
- Additional operations like `reduce` (for dimension reduction)
- Support for einsum operations with the same intuitive notation
- Expanded backends for PyTorch, TensorFlow, and JAX

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the original [einops](https://github.com/arogozhnikov/einops) library by Alex Rogozhnikov
- Thanks to the NumPy community for providing the foundation for scientific computing in Python
