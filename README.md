# einflux

![einflux logo](https://img.shields.io/badge/einflux-reimagining%20einops-blue)
[![PyPI version](https://img.shields.io/badge/pypi-coming%20soon-orange)](https://github.com/Kkumar-007/einflux)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**einflux** is a Python library that provides a ground-up reimplementation of the popular [einops](https://github.com/arogozhnikov/einops) library, offering tensor operations using the Einstein-inspired notation pattern for NumPy arrays. This library aims to make NumPy array manipulations more intuitive, readable, and less error-prone.

## What is einflux?

einflux provides a clean, intuitive API for tensor manipulations specifically for NumPy arrays. It uses a consistent notation pattern to express complex tensor operations that would otherwise require multiple steps with traditional NumPy operations like reshape, transpose, etc.

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

## Key Features

- **NumPy-focused implementation**: Built specifically for NumPy array operations
- **Intuitive Einstein-inspired notation**: Express complex tensor operations in a single, readable line
- **Error checking**: Validate array shapes against patterns at runtime
- **Composition**: Compose multiple operations seamlessly
- **Pure Python implementation**: Built from scratch with minimal dependencies (only NumPy)

## Core Design Philosophy

einflux was built with the following principles in mind:

1. **Readability First**: Operations should be self-documenting and express intent
2. **Safety**: Runtime shape validation to catch errors early
3. **Performance**: Optimized NumPy operations that maintain efficiency
4. **Simplicity**: Focus on NumPy allows for a leaner, more focused implementation

## Implementation Approach

### Parser Architecture

The core of einflux is a robust Python parser that:

1. Parses the Einstein-inspired notation patterns
2. Extracts dimensions and their relationships
3. Generates the appropriate low-level NumPy operations

The parser follows these steps:
- Tokenizes the input and output patterns
- Identifies named dimensions, anonymous dimensions, and reduction operations
- Builds a dimension mapping between input and output
- Validates shape consistency with the actual NumPy array

### NumPy Backend Design

einflux is designed specifically for NumPy operations:

```
├── einflux/
│   ├── parser.py
│   ├── operations.py
│   ├── utils.py
│   └── numpy_backend.py
```

The implementation leverages NumPy's efficient array operations while providing a more intuitive interface through Einstein-inspired notation.

## Core Operations

### `rearrange`

Rearranges NumPy array dimensions according to the pattern.

```python
# Flatten image dimensions
rearrange(x, 'b h w c -> b (h w) c')

# Matrix transposition
rearrange(x, 'i j -> j i')

# Reshape embeddings
rearrange(x, 'b n (h d) -> b n h d', h=8)
```

### `reduce`

Reduces NumPy array dimensions according to the pattern.

```python
# Global average pooling
reduce(x, 'b h w c -> b c', 'mean')

# Sum over batch dimension
reduce(x, 'b h w c -> h w c', 'sum')
```

### `repeat`

Repeats NumPy array dimensions according to the pattern.

```python
# Repeat embedding for each position
repeat(x, 'b c -> b n c', n=100)

# Broadcasting a vector to a matrix
repeat(x, 'i -> i j', j=28)
```

### `einsum`

Einstein summation with einflux's notation system, built on top of `numpy.einsum`.

```python
# Matrix multiplication
einsum('b i k, b k j -> b i j', x, y)

# Batch-wise dot product
einsum('b n d, b d -> b n', x, y)
```

## Performance Considerations

einflux prioritizes both readability and performance:

1. **Optimized NumPy Operations**: Leverages NumPy's efficient vectorized operations
2. **Composition Optimization**: When operations are composed, they are analyzed to reduce intermediate allocations
3. **Minimal Overhead**: Thin wrapper around NumPy functionality to maintain performance
4. **Vectorized Operations**: Uses NumPy vectorization where possible for performance

## Examples

### Image Processing

```python
# Reshape image batch for convolution
x = rearrange(images, 'b h w c -> b c h w')

# Patch extraction
patches = rearrange(images, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=8, p2=8)
```

### Data Manipulation

```python
# Flatten selected dimensions
flattened = rearrange(data, 'b c h w -> b (c h w)')

# Reshaping for batch processing
batched = rearrange(data, '(b n) ... -> b n ...', b=8)
```

### Scientific Computing

```python
# Tensor factorization
factors = rearrange(tensor, 'i j k -> i (j k)')

# Matrix operations
result = einsum('i j, j k -> i k', matrix_a, matrix_b)
```

## Python Version Support

einflux is compatible with Python 3.7 and above, and requires NumPy 1.19+. The library is tested with the following Python versions:

- Python 3.7
- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Future Development

While einflux currently focuses exclusively on NumPy arrays, future versions may include support for:

- PyTorch tensors
- TensorFlow tensors
- JAX arrays

## Acknowledgments

- Inspired by the original [einops](https://github.com/arogozhnikov/einops) library by Alex Rogozhnikov
- Thanks to all contributors and the NumPy community for feedback and support

## Citation

If you use einflux in your research, please cite:

```bibtex
@software{kumar2025einflux,
  author = {Kumar, K},
  title = {einflux: A NumPy-focused Reimplementation of einops},
  year = {2025},
  url = {https://github.com/Kkumar-007/einflux}
}
```
