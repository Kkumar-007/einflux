# einflux
🌀 einflux – Because tensors deserve a little chaos.

> _Effortlessly reshape reality... or at least your tensors._

**einflux** is a chaotic-good tensor rearrangement library inspired by [`einops`](https://github.com/arogozhnikov/einops). It's fast, flexible, and built for people who think dimension manipulation should feel more like casting spells than writing code.

---

## Aim

I tried to **re-implement `einops` from scratch**, and... mission accomplished!  
The core idea was to rebuild the functionality, parsing, and reshaping logic from the ground up using just **NumPy**.

> While this version works great and supports a wide range of rearrangement patterns, benchmarks show that it's currently **~2.42x slower** than the original `einops` library. Totally expected for a first version – and hey, it's readable!

---

## Features

- **Pattern-based dimension sorcery** – Write rearrange logic as readable strings like `"a b c -> c a b"`
- **Split & merge dimensions** – Do `(a b) c -> a (b c)` like it's nothing
- **Wildcard & ellipsis support** – Use `*` or `...` to match variable dims
- **Smart dimension inference** – Forget hardcoding sizes
- **Fast & memory-friendly** – Powered by NumPy’s vectorized magic
- **Solid error handling** – Clear feedback when patterns fail

---

## Pattern Syntax Cheat Sheet

```python
"a b c -> c a b"       # Transpose
"(a b) c -> a (b c)"   # Split then merge
"a * b -> b * a"       # Wildcard match
"a ... b -> b ... a"   # Ellipsis life
"(a 2) b -> b (2 a)"   # Fixed numeric dimensions
```

---

## Design Decisions

- **One-pass parser for patterns** – fast and less annoying
- **Inference engine that auto-fills unknown dims** - like a mind-reader
- **Memory efficient reshapes/transposes** – no unnecessary copies
- **Errors that make sense** – to provide efficient feedback

---

## How to Run
### Basic Usage
```python
import numpy as np
from einflux import rearrange

x = np.zeros((2, 3, 4))

# Rearrange example
result = rearrange(x, "a b c -> c a b")
print(result.shape)  # (4, 2, 3)

# Splitting dimensions
x = np.zeros((6, 4))
result = rearrange(x, "(a b) c -> a (b c)", a=2, b=3)
print(result.shape)  # (2, 12)
```

### Running Tests
Make sure pyytest is installed.
```bash
pip install pytest
```

```python
import pytest

pytest tests.py -v
```

---

## Performance Vibes
- Built with NumPy for speed
- Pattern parsing optimized in a single-pass
- Efficient dimension inference without slowing you down
- Clean error handling

## Benchmark Report:
I ran performance comparisons between einflux and the original einops library across various tensor sizes and rearrangement patterns.

On average, einflux is ~2.42x slower than einops.

This performance difference is expected due to the additional parsing logic and lack of low-level optimization in the current prototype. Future versions aim to narrow this gap while maintaining readability and flexibility.

📝 The full performance report is included in performance_report.md

---

## Edge Cases? We Handle Those Too
- ✅ Repeated dimensions in output like a b -> a b a
- ✅ Mixing numbers and names (a 2) b -> b (2 a)
- ✅ Wildcards and ellipses like a champ
- ✅ Smart validation that tells you what’s wrong

---

## 📦 Installation
Not on PyPI yet. Just clone this:

```bash
git clone https://github.com/Kkumar-007/einflux.git
cd einflux
```

---

## 💬 Contributing
Got ideas? Bugs? Cool pattern syntax? Open a PR or start an issue. Tensor wizards welcom
