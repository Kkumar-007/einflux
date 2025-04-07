# einflux
🌀 einflux – Because tensors deserve a little chaos.

> _Effortlessly reshape reality... or at least your tensors._

**einflux** is a chaotic-good tensor rearrangement library inspired by [`einops`](https://github.com/arogozhnikov/einops). It's fast, flexible, and built for people who think dimension manipulation should feel more like casting spells than writing code.

---

## ✨ Features

- 🔁 **Pattern-based dimension sorcery** – Write rearrange logic as readable strings like `"a b c -> c a b"`
- 🧩 **Split & merge dimensions** – Do `(a b) c -> a (b c)` like it's nothing
- 🎭 **Wildcard & ellipsis support** – Use `*` or `...` to match variable dims
- 🔍 **Smart dimension inference** – Forget hardcoding sizes, we gotchu
- ⚡ **Fast & memory-friendly** – Powered by NumPy’s vectorized magic
- 🧼 **Solid error handling** – Clear feedback when patterns go 💥

---

## 🧠 Pattern Syntax Cheat Sheet

```python
"a b c -> c a b"       # Transpose
"(a b) c -> a (b c)"   # Split then merge
"a * b -> b * a"       # Wildcard match
"a ... b -> b ... a"   # Ellipsis life
"(a 2) b -> b (2 a)"   # Fixed numeric dimensions

