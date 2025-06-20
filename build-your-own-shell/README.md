# Build Your Own Shell

**Project Goal:** Create a basic Unix shell interface, useful for HPC simulation workflows.

## Structure
- `src/` – source code
- `assets/` – screenshots or architecture diagrams
- `NOTES.md` – optional reflections and key learnings
# Build Your Own Neural Network (From Scratch)

This project demonstrates how to implement a **feedforward neural network from scratch** using only NumPy — no libraries like PyTorch or TensorFlow. It's designed as both a learning exercise and a stepping stone toward building real-world medical AI systems.

---

## What It Does

- Solves the XOR problem using a 2-layer neural net
- Implements forward propagation, backpropagation, and gradient descent manually
- Uses sigmoid activation to learn non-linear relationships
- Fully explained in **`main_deep_annotated.py`** with clinical parallels

---

## Clinical Learning Parallel

**Real Healthcare Scenario:**  
You’re building a system that predicts adverse patient outcomes (e.g. brain death, cardiac arrest, transplant failure) from 2–10 patient vitals.

This code teaches you:
- How vitals can be weighted and transformed by neurons
- How non-linearity helps discover hidden interactions
- What backpropagation would look like inside a digital twin or predictive monitoring device

---

## Files

- `src/main.py`: Clean XOR model without comments
- `src/main_annotated.py`: Lightly commented version
- `src/main_deep_annotated.py`: Full walkthrough + healthcare integration notes
- `assets/`: Place for visual architecture diagrams (optional)
- `NOTES.md`: Learning reflections (fill out as you go)

---

## Run It

```bash
cd src/
python main_deep_annotated.py
```

---

## What's Next?

- Replace XOR with synthetic health data (e.g., HR, BP)
- Train it to predict ischemia (like in digital twin)
- Add softmax to support multiple outcomes (e.g., low / medium / high risk)
- Expand to multi-layer networks

---

## Author

**Ihunna Amugo**  
DDS Candidate | MHA | MS | REHS | PhD(c) Computational Engineering  
GitHub: [@ihunnamatata](https://github.com/ihunnamatata)
