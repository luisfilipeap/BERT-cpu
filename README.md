# BERT-CPU

> *"What I cannot create, I do not understand."*  
> — Richard Feynman

This project is built in that spirit: the surest way to understand BERT is to
build it from scratch. Here, *from scratch* is meant literally — **NumPy is the
only dependency, used only as an array computation backend**.

## What is BERT-cpu?

BERT-cpu is a NumPy-only, CPU-only environment for learning how a BERT-style
model is built and trained *from the autograd level up* — no GPUs, no CUDA, no
heavyweight framework. It favours **clarity, reproducibility and
inspectability** over raw performance. **Our engine is deliberately not meant to compete with PyTorch, TensorFlow, JAX or any other Deep Learning framework**.

Its real strength is that it makes the **learning machinery visible**. For
example, the didactic walkthrough turns the autograd engine inside out: it draws
the computational graph **vertically, the way `git log --graph` stacks commits**
(output on top, operands below), shows every node's forward value, then
**animates backpropagation step by step** — filling in each gradient with the
exact chain-rule formula used, all the way from the output seed back to the
inputs. The maths of training a network stops being a black box and becomes
something you can read line by line:

![Gradient-graph walkthrough](docs/demo.svg)

The same drawing scales up unchanged: a single `wᵀ @ x` op, or a whole stack of
`Linear` layers — it is always *a bigger graph of the same kind* (run
`python -m learn.viz_nn` to see three activated layers chained together).

## Who is it for?

### For students

- **A democratic way to learn the maths.** See the full mathematics of training
  a network on any laptop — no GPU, CUDA, or cloud budget required. **The barrier to understanding deep learning becomes curiosity, not expensive hardware: the maths is the point, the hardware is not**.
- Study how a network is differentiated **from first principles**, seeing how a   `Tensor` stores its value, gradient, parent nodes, op label, and local   backward function.
- Watch the **computational graph** get built during the forward pass, and flow backward, one node at a time, with the chain-rule formula shown at each step.
- Connect mathematical formulas directly to executable NumPy code.

### For teaching

- Use the step-by-step walkthrough as a lecture artifact for courses on
  automatic differentiation, deep-learning fundamentals, and (as the higher
  layers land) Transformer models.
- Demonstrate each engine operation independently before assembling larger
  computations.
- Build exercises where students modify a single op or backward rule and
  immediately observe the effect on the gradients.
- Debug student implementations by comparing **analytic vs numerical**
  gradients.
- Choose the precision and RNG seed of the demo on the command line, so the
  same walkthrough can be shown under different settings
  (`python -m learn.viz_engine --precision float32 --seed 0`).

### For researchers

- Validate that an idea is **mathematically and computationally sound** in a
  transparent setting before investing in a large-scale implementation.
- **Reproducibility:** one global seed (`bert_cpu.set_seed(0)`) makes a whole run
  repeatable end to end, with far fewer sources of randomness than large-scale
  GPU training.
- **Minimal stack:** only NumPy — no CUDA versions, GPU kernels, or distributed
  setup to reproduce, so experiments are easy to replicate on common hardware.
  Skip the infrastructure tax. Test fundamental "first-principles" hypotheses
  (e.g., alternative credit assignment, sparse topologies) instantly on **any machine**,
  without managing CUDA toolkits, dependency hell, or cluster allocations.
- **Full inspectability:** the entire computational path (values *and*
  gradients) is visible, which makes failures and learning dynamics easy to
  examine in small, controlled toy settings.
- **Precision control:** run the engine in float64 (stable, default) or
  float32/float16 to study the numerical behaviour of an idea.

## Current status

The project is being built bottom-up. What exists today:

- **Autograd engine** (`bert_cpu/engine.py`) — ✅ complete. The `Tensor` class
  with reverse-mode `backward()`, broadcasting, elementwise ops, `@` (matmul),
  activations (`tanh`, `relu`, `gelu`), reductions (`sum`, `mean`, `var`,
  `max`), `softmax`, indexing, `cat`, plus `set_seed` and precision control.
- **NN layers** (`bert_cpu/nn.py`) — 🚧 in progress. `Module` (parameter
  collection, `zero_grad`, `train`/`eval`), `Parameter`, `Linear` (with the bias
  folded into the weight via the `x_0 = 1` trick), and `xavier_uniform` are
  implemented. `Embedding`, `LayerNorm`, `Dropout`, `Sequential` are scaffolded.
- **Visualisations** (`learn/`) — ✅ `viz_engine` (the autograd engine) and
  `viz_nn` (a stack of `Linear` layers and its chain rule).
- **Higher layers** — ⏳ scaffolded only: attention (`attention.py`), the
  Transformer encoder (`transformer.py`), optimizers (`optim.py`), losses
  (`loss.py`) and the tokenizer (`tokenizer.py`) still raise
  `NotImplementedError`.

## Requirements

- Python >= 3.8
- NumPy (the only runtime dependency)

## Setup

Create a virtual environment and install the dependencies:

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows (PowerShell)

# Install the dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** if `python3 -m venv` fails with an `ensurepip is not available`
> error, your interpreter is missing the `venv` module. On Debian/Ubuntu install
> it with `apt install python3-venv`, or use a `pyenv`-managed Python.

## Learning path (start here)

This project is meant to be *read and run* from the ground up. Everything BERT
does eventually reduces to one idea: a graph of tensor operations through which
gradients flow backward. So the very first thing to understand — right after
installing — is **how the autograd engine works**.

**Step 1 — see the gradient engine in action.** The didactic visualisations live
in the `learn/` package and are meant to be run directly (use `-m` from the
project root so `bert_cpu` is importable — running the path directly is not).
With the virtual environment activated:

```bash
python -m learn.viz_engine
```

This standalone run also lets you choose the numerical precision and the RNG
seed, so you can watch the same walkthrough under different settings and get
reproducible numbers:

```bash
python -m learn.viz_engine --precision float32 --seed 0
python -m learn.viz_engine --precision float16 --seed 42
```

- `--precision` picks the engine's float dtype (`float16` / `float32` /
  `float64`; default `float64`).
- `--seed` seeds NumPy's RNG so the run is reproducible (omit for a random run).

What you will see, and what to take away from it:

1. **An input column vector** `x` and a **weight column vector** `w`. The demo
   uses the "bias trick": the input is *augmented* with a leading constant
   `x_0 = 1`, so `w_0` plays the role of the bias.
2. **The forward pass** of a tiny linear layer, computing `z = wᵀ @ x` (a matrix
   multiplication) and then `y = tanh(z)`, step by step.
3. **The computational graph as a vertical tree** (output on top, operands
   hanging below with `git`-style connectors), each node annotated with its
   forward `value` and its `grad` — column vectors and matrices drawn in their
   mathematical shape.
4. **The backward pass, animated step by step.** Each step fills in one node's
   gradient and prints the local derivative rule it used (for the `tanh`, the
   matmul `@`, and the transpose), so you watch the chain rule propagate from
   the output seed back to every input.

Read the graph from the top down to follow the forward pass, then read the
gradients to see how `backward()` distributes the chain rule from the output
back to every input.

**Step 2 — see a layer, and how stacking layers chains derivatives.** Once the
engine clicks, move up one level to the `nn` layers:

```bash
python -m learn.viz_nn --seed 5      # seed 5 keeps every ReLU unit lively
```

This walkthrough shows that a `Linear` layer is **just a matrix** (with the bias
folded into row 0 via the same `x_0 = 1` trick), then a nonlinearity; and that
**stacking three activated layers turns the backward pass into a chain of
derivatives**, propagated layer by layer and multiplied at each step by the
activation slope `act'(z)` and the weight matrix. The rest of the library
(attention, the full encoder) is just *bigger graphs of the same kind*.

**Step 3 — confirm everything is correct.** Run the software tests
(broadcasting, matmul, softmax, finite-difference gradient checks, the layers,
and the cross-layer chain rule):

```bash
pytest
```

From here you are ready to explore the higher-level modules. The full testing
reference is in [Tests and didactic walkthroughs](#tests-and-didactic-walkthroughs).

## Usage

With the virtual environment activated, import the library from the project
root. The package directory is `bert_cpu` (underscore — the importable name),
while the distribution is named `bert-cpu`.

**The autograd engine.** Build an expression out of `Tensor`s and differentiate
it with `backward()`:

```python
from bert_cpu import engine as cpu

cpu.set_seed(0)                        # reproducible run

x = cpu.Tensor([[2.0], [-3.0]])        # input column vector (2, 1)
w = cpu.Tensor([[0.5], [1.5]])         # weight column vector (2, 1)
y = (w.T @ x).tanh()                   # forward pass: a tiny linear unit

y.backward()                           # reverse-mode autodiff
print(x.grad, w.grad)                  # gradients dy/dx, dy/dw
```

**An `nn.Linear` layer.** The first building block on top of the engine is in
place. It uses the column-vector "bias trick" (input augmented with `x_0 = 1`,
bias folded into row 0 of the weight), so a layer is just `Wᵀ @ x`:

```python
from bert_cpu import engine as cpu, nn

cpu.set_seed(0)

x = cpu.Tensor([[2.0], [-3.0]])        # input column vector (in_features, batch)
layer = nn.Linear(2, 4)                # weight is (in_features + 1, out_features)
y = layer(x).relu()                    # forward: relu(Wᵀ @ [1; x])

y.sum().backward()                     # gradients flow into layer.weight
print(layer.weight.grad)               # dL/dW (bias row included)
```

> The higher-level pieces (`BERTModel`, `MultiHeadAttention`, optimizers, losses,
> tokenizer) are scaffolded but not implemented yet — see
> [Current status](#current-status) above.

## Tests and didactic walkthroughs

The project keeps two concerns cleanly apart:

1. **Software tests** (`test/`) — ordinary correctness checks (broadcasting,
   matmul, softmax, finite-difference gradient checks, the layers, the
   cross-layer chain rule). These only assert; they do not teach.
2. **Didactic walkthroughs** (`learn/`) — runnable visualisations that *print to
   the console* to teach what is happening internally, e.g. drawing the
   computational graph and animating how reverse-mode autodiff propagates the
   chain rule from the output back to the inputs.

First install the test dependency (already covered if you ran the setup above):

```bash
pip install pytest
```

### Run the software tests

```bash
pytest                       # run every software test
pytest -v                    # verbose, one line per test
pytest test/test_engine.py   # just the autograd-engine tests
pytest test/test_nn.py       # just the nn-layer tests
```

### Run the didactic walkthroughs

The walkthroughs live in `learn/` and are run as modules from the project root
(use `-m` so `bert_cpu` is importable, not by path):

```bash
python -m learn.viz_engine                               # autograd engine, node by node
python -m learn.viz_engine --precision float32 --seed 0  # pick dtype and seed
python -m learn.viz_nn --seed 5                           # a layer + the chain rule over a stack
```

- `viz_engine` — input vector → forward pass → ASCII graph annotated with
  gradients → step-by-step backward animation.
- `viz_nn` — a `Linear` layer as a matrix, then three activated layers whose
  backward pass is shown as a layer-by-layer chain of derivatives.

Both accept `--precision` and `--seed`. (A lightweight `pytest test/test_viz.py`
just smoke-checks that the walkthroughs still run.)

> New here? Follow the [Learning path](#learning-path-start-here) above — it
> walks you through the engine demo first, since every other module is built
> on top of it.
