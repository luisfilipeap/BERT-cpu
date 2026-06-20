"""Didactic visualisation: a linear layer is a matrix, and stacking layers
builds a chain of derivatives.

Where ``viz_engine.py`` shows a *single* node-level graph, this walkthrough
moves up one level of abstraction and shows what ``nn.py`` is really about:

1. a **linear layer is just a matrix** (with the bias folded in as row 0), then
   a nonlinearity — ``a = act(Wᵀ @ x)``; and
2. when several such layers are **stacked**, each with its own nonlinear
   activation, the backward pass becomes a **chain of derivatives**: the
   gradient is propagated layer by layer, multiplied at every step by that
   layer's activation slope ``act'(z)`` and its weight matrix.

It is *not* a test: the layers' correctness is checked in ``test/test_nn.py``.
This file is here to be **read and run**::

    python -m learn.viz_nn                               # defaults: float64, random seed
    python -m learn.viz_nn --precision float32 --seed 5  # pick dtype and seed
"""

import numpy as np

from bert_cpu import engine as cpu
from bert_cpu import nn
from learn.console import console
from learn.viz_engine import draw_graph


def _act_derivative(name: str, z: np.ndarray) -> np.ndarray:
    """Return ``act'(z)``, the activation's local slope, computed by the engine.

    Rather than hand-coding each derivative here, we let the autograd engine do
    it: for an elementwise activation, ``act(z).sum().backward()`` leaves
    ``z.grad[i] = act'(z_i)``. So this returns the *same* local slope the engine
    uses inside its backward pass — the very factor the chain rule multiplies by.
    """
    t = cpu.Tensor(z.copy())
    getattr(t, name)().sum().backward()
    return t.grad


_LAYER_HEADERS = ["layer", "z = Wᵀ@x", "a = act(z)", "dL/dW"]


def _draw_layer_table(rows: list) -> None:
    """Print a ``layer | z | a | dL/dW`` table, one row per layer (output on top).

    This is the layer-level analogue of ``draw_graph`` in ``viz_engine.py``:
    instead of one row per *node*, there is one row per *layer*, showing its
    pre-activation ``z = Wᵀ @ x``, its activation ``a = act(z)`` and (once the
    backward pass has reached it) the gradient ``dL/dW`` of the loss w.r.t. that
    layer's weight. ``dL/dW`` is a matrix, so it is drawn as a stacked table (the
    row grows to as many lines as the matrix has rows); ``z`` and ``a`` are
    vectors and stay on one line. A ``-`` means the chain rule has not reached
    that layer yet, so redrawing with more filled-in rows animates the backward
    pass from the output downwards.
    """
    print()
    print(console.format_table(
        rows,
        headers=_LAYER_HEADERS,
        col_colors=[console.math, console.value, console.value, console.value],
    ))
    print()


def demo_linear_chain() -> None:
    """Visualise the chain rule over a stack of three activated linear layers.

    Stacks three ``nn.Linear`` layers, each with a different nonlinear activation
    (ReLU → tanh → GELU), draws the whole stack as one computational graph, then
    replays the backward pass **one layer at a time**, spelling out the
    chain-rule factors that connect the loss to every weight.
    """
    print("\n")
    print(console.text("=" * 70))
    print(console.text("NN demo:  stacking Linear layers and computing its chain of derivatives"))
    print(console.text("=" * 70))

    print(console.text("\nThe network consists of three fully connected layers. In each layer,\nthe affine transformation ")
          + console.math("Wᵀ @ [1; ·]") 
          + console.text(", where the leading ") + console.math("1") + console.text(" accounts for\nthe bias term, is followed by a nonlinear activation function."))
    
    print(console.text("\nRead the set of equations describing the network from bottom to top, \nfollowing the forward pass from the input ")
          + console.math("x") + console.text(" to the prediction ") + console.math("ŷ") + console.text(":\n"))
    
    print(console.math("  ŷ  = gelu(W3ᵀ @ [1; a2])"))
    print(console.math("  a2 = tanh(W2ᵀ @ [1; a1])"))
    print(console.math("  a1 = relu(W1ᵀ @ [1; x ])"))

    x = cpu.Tensor(np.random.randn(2, 1));  x.label = "x"   # column input (2, 1)
    layer1 = nn.Linear(2, 2);  layer1.weight.label = "W1"
    layer2 = nn.Linear(2, 2);  layer2.weight.label = "W2"
    layer3 = nn.Linear(2, 1);  layer3.weight.label = "W3"

    z1 = layer1(x);   z1.label = "z1";   a1 = z1.relu();  a1.label = "a1"
    z2 = layer2(a1);  z2.label = "z2";   a2 = z2.tanh();  a2.label = "a2"
    z3 = layer3(a2);  z3.label = "z3";   a3 = z3.gelu();  a3.label = "ŷ"
    loss = a3.sum();  loss.label = "L"

    # The three weight matrices ARE the network: together they hold every
    # parameter (row 0 of each is that layer's bias). Show them, and the input.
    print(console.text("\nThe network is its three weight matrices ") + console.math("W1, W2, W3")
          + console.text(" (first row of each is\nthe neuron's bias in that layer). Together they hold every parameter the network has:\n"))
    for w in (layer1.weight, layer2.weight, layer3.weight):
        print(console.math(f"  {w.label}  (shape {w.shape[0]}x{w.shape[1]}) ="))
        print(console.fmt_matrix(w.data, indent="      "))

    print(console.text("\nThe input is the column vector ") + console.math("x (shape 2x1)\n"))
    print(console.fmt_matrix(x.data, indent="      "))

    def show_matmul(eq: str, weight, prev_data: np.ndarray, z) -> None:
        """Print ``z = Wᵀ @ [1; prev]`` as actual matrices: Wᵀ, the augmented
        input column ``[1; prev]``, and the resulting ``z`` — side by side."""
        aug = np.vstack([np.ones((1, prev_data.shape[1])), prev_data])  # prepend x_0 = 1
        print(console.math("  " + eq + " ="))
        block = console.hjoin([
            (console.fmt_matrix_plain(weight.data.T), console.value),   # Wᵀ
            ("@", console.math),
            (console.fmt_matrix_plain(aug), console.value),             # [1; prev]
            ("=", console.math),
            (console.fmt_matrix_plain(z.data), console.value),          # z
        ])
        print("\n".join("      " + ln for ln in block.split("\n")))

    print(console.text("\nThe ")+console.label("Forward pass")+console.text(": each pre-activation ") + console.math("z")
          + console.text(" is a matrix product ") + console.math("Wᵀ @ [1; ·]")
          + console.text(", then an activation:\n"))
    
    show_matmul("z1 = W1ᵀ @ [1; x ]", layer1.weight, x.data, z1)
    console.kv("  a1 = relu(z1)        = ", console.fmt_auto(a1.data), color=console.math)
    show_matmul("z2 = W2ᵀ @ [1; a1]", layer2.weight, a1.data, z2)
    console.kv("  a2 = tanh(z2)        = ", console.fmt_auto(a2.data), color=console.math)
    show_matmul("z3 = W3ᵀ @ [1; a2]", layer3.weight, a2.data, z3)
    console.kv("  ŷ  = gelu(z3)        = ", console.fmt_auto(a3.data), color=console.math)
    console.kv("  L  = sum(ŷ)          = ", console.fmt_auto(loss.data), "    (a scalar to differentiate)", color=console.math)

    print(console.text("\nThose six steps are really one ") + console.label("computational graph")
          + console.text(". Here is the whole stack —\nthe output ") + console.math("L")
          + console.text(" at the top, down through every op to the inputs ") + console.math("x")
          + console.text(" and\nthe weights ") + console.math("W1, W2, W3")
          + console.text(" (each ") + console.math("Linear") + console.text(" is a ")
          + console.math("cat") + console.text(" + ") + console.math("@") + console.text("):"))
    draw_graph(loss, known=set())
    print(console.text("It is exactly the engine's graph from ") + console.math("viz_engine")
          + console.text(", just bigger: three ") + console.math("Wᵀ @ [1; ·]")
          + console.text(" blocks chained through their activations."))

    # Run the real autograd; the narration below reads the engine's gradients.
    for p in (x, layer1.weight, layer2.weight, layer3.weight):
        p.zero_grad()
    loss.backward()

    # output-on-top ordering, like draw_graph
    specs = [
        ("layer 3", layer3, z3, a3, "gelu", a2, "a2"),
        ("layer 2", layer2, z2, a2, "tanh", a1, "a1"),
        ("layer 1", layer1, z1, a1, "relu", x, "x"),
    ]

    def make_rows(known: set) -> list:
        rows = []
        for name, layer, z, a, _act, _prev, _pn in specs:
            # Everything is shown in its mathematical shape: z and a as column
            # vectors, dL/dW as a matrix (all stacked by ``format_table``).
            g = console.fmt_auto_plain(layer.weight.grad) if name in known else "-"
            rows.append([name, console.fmt_auto_plain(z.data), console.fmt_auto_plain(a.data), g])
        return rows

    print(console.text("\nBackprop seeds the output, ") + console.math("dL/dŷ = dL/dL · dL/dŷ = 1")
          + console.text(", then walks back layer by layer.\nEach ") + console.label("dL/dW")
          + console.text(" is still '-' until the chain rule reaches its layer:"))
    known: set = set()
    _draw_layer_table(make_rows(known))

    for step, (name, layer, z, a, act, prev, prev_name) in enumerate(specs, start=1):
        slope = _act_derivative(act, z.data)            # act'(z), the local slope
        upstream = a.grad                               # dL/da  (arriving at this layer)
        dz = z.grad                                     # dL/dz = dL/da ⊙ act'(z)
        dW = layer.weight.grad                          # dL/dW = [1; a_prev] @ (dL/dz)ᵀ
        dprev = prev.grad                               # dL/d(a_prev) = W[1:] @ dL/dz

        print(console.label(f"\nSTEP {step}") + console.text(": backprop through ")
              + console.math(name) + console.text("  (activation ") + console.math(act) + console.text(")"))
        # Every quantity is shown in its mathematical shape: column vectors stack
        # vertically and the weight gradient is a matrix (``kv`` drops multi-line
        # values onto the lines below the label).
        console.kv("        upstream            dL/d" + a.label + " = ", console.fmt_auto(upstream), color=console.math)
        console.kv("        activation slope    " + act + "'(" + z.label + ") = ", console.fmt_auto(slope),
                   "   [from the engine]", color=console.math)
        console.kv("     -> pre-activation grad  dL/d" + z.label + " = dL/d" + a.label + " ⊙ " + act + "'(" + z.label + ") = ",
                   console.fmt_auto(dz), color=console.math)
        console.kv("        weight grad         dL/dW = [1; " + prev_name + "] @ (dL/d" + z.label + ")ᵀ = ",
                   console.fmt_auto(dW), color=console.math)
        console.kv("        propagate back      dL/d" + prev_name + " = W[1:] @ dL/d" + z.label + " = ",
                   console.fmt_auto(dprev), "   (the bias row is dropped)", color=console.math)

        known.add(name)
        _draw_layer_table(make_rows(known))

    print(console.text("Each step multiplied the incoming gradient by two local factors — the\nactivation slope ")
          + console.math("act'(z)") + console.text(" and the weight matrix ") + console.math("W")
          + console.text(". Composing those\nfactors from the output back to the input ") + console.text("is")
          + console.text(" the chain rule; ") + console.math("backward()")
          + console.text(" just\nautomates it over the whole stack."))


# ====================================================================== #
# Standalone entry point (mirrors viz_engine.main)
# ====================================================================== #
def main(argv=None) -> None:
    """Run the layer/chain walkthrough with a chosen precision and RNG seed.

    The precision and seed are configured on the engine *before* anything is
    built, so they govern the whole run::

        python -m learn.viz_nn --precision float32 --seed 5
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision",
        default="float64",
        help="NumPy float dtype for the engine (e.g. float16, float32, float64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the NumPy RNG (reproducible run). Omit for a random run. "
             "Seed 5 is a good lively choice (all ReLU units fire).",
    )
    args = parser.parse_args(argv)

    try:
        dtype = np.dtype(args.precision)
    except TypeError as exc:
        parser.error(f"unknown precision {args.precision!r}: {exc}")
    if not np.issubdtype(dtype, np.floating):
        parser.error(f"precision must be a floating type, got {args.precision!r}")

    cpu.default_dtype = dtype.type
    if args.seed is not None:
        cpu.set_seed(args.seed)

    seed_str = "random" if args.seed is None else str(args.seed)
    print(console.text("Engine config: ") + console.label("seed") + console.text("=")
          + console.value(seed_str) + console.text("  ") + console.label("precision")
          + console.text("=") + console.value(dtype.name))
    demo_linear_chain()


if __name__ == "__main__":
    main()
