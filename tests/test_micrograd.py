"""
Tests for the micrograd Value engine and MLP components.

Validates forward pass computations, automatic backpropagation gradients,
and neural network training convergence by cross-checking against PyTorch.
"""

import math
import random
import pytest


# ---------------------------------------------------------------------------
# Value class (extracted from the lecture notebooks)
# ---------------------------------------------------------------------------

class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# ---------------------------------------------------------------------------
# Neural network classes (extracted from the lecture notebooks)
# ---------------------------------------------------------------------------

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ===== Tests ================================================================

# -- Value: forward pass ----------------------------------------------------

class TestValueForward:

    def test_add(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert c.data == 5.0

    def test_mul(self):
        a = Value(2.0)
        b = Value(-3.0)
        c = a * b
        assert c.data == -6.0

    def test_pow(self):
        a = Value(3.0)
        c = a ** 2
        assert c.data == 9.0

    def test_neg(self):
        a = Value(4.0)
        assert (-a).data == -4.0

    def test_sub(self):
        a = Value(5.0)
        b = Value(3.0)
        assert (a - b).data == 2.0

    def test_div(self):
        a = Value(6.0)
        b = Value(3.0)
        assert (a / b).data == pytest.approx(2.0)

    def test_tanh(self):
        a = Value(0.0)
        assert a.tanh().data == pytest.approx(0.0)
        b = Value(1.0)
        assert b.tanh().data == pytest.approx(math.tanh(1.0))

    def test_exp(self):
        a = Value(1.0)
        assert a.exp().data == pytest.approx(math.e)

    def test_rmul(self):
        a = Value(3.0)
        c = 2 * a  # triggers __rmul__
        assert c.data == 6.0

    def test_radd(self):
        a = Value(3.0)
        c = 1 + a  # triggers __radd__
        assert c.data == 4.0

    def test_composite_expression(self):
        """L = (a * b + c) * f  from the lecture."""
        a = Value(2.0)
        b = Value(-3.0)
        c = Value(10.0)
        f = Value(-2.0)
        L = (a * b + c) * f
        assert L.data == -8.0


# -- Value: backward pass ---------------------------------------------------

class TestValueBackward:

    def test_simple_product_grad(self):
        """L = (a*b + c) * f  — check all gradients."""
        a = Value(2.0)
        b = Value(-3.0)
        c = Value(10.0)
        f = Value(-2.0)
        e = a * b
        d = e + c
        L = d * f
        L.backward()

        assert L.data == -8.0
        assert f.grad == pytest.approx(4.0)   # dL/df = d = a*b+c = 4
        assert d.grad == pytest.approx(-2.0)  # dL/dd = f = -2
        assert c.grad == pytest.approx(-2.0)  # dL/dc = dL/dd * dd/dc = -2
        assert e.grad == pytest.approx(-2.0)  # dL/de = dL/dd * dd/de = -2
        assert a.grad == pytest.approx(6.0)   # dL/da = dL/de * b = -2*-3 = 6
        assert b.grad == pytest.approx(-4.0)  # dL/db = dL/de * a = -2*2 = -4

    def test_neuron_backward(self):
        """Single neuron: o = tanh(x1*w1 + x2*w2 + b)."""
        x1 = Value(2.0)
        x2 = Value(0.0)
        w1 = Value(-3.0)
        w2 = Value(1.0)
        b = Value(6.8813735870195432)
        n = x1 * w1 + x2 * w2 + b
        o = n.tanh()
        o.backward()

        assert o.data == pytest.approx(0.7071067811865476, abs=1e-6)
        # tanh'(n) = 1 - tanh(n)^2 ≈ 0.5
        assert n.grad == pytest.approx(0.5, abs=1e-4)
        assert x1.grad == pytest.approx(-1.5, abs=1e-3)
        assert w1.grad == pytest.approx(1.0, abs=1e-3)

    def test_add_same_variable(self):
        """b = a + a  — gradient should accumulate to 2."""
        a = Value(3.0)
        b = a + a
        b.backward()
        assert a.grad == pytest.approx(2.0)

    def test_mul_same_variable(self):
        """b = a * a  — gradient should be 2*a = 6."""
        a = Value(3.0)
        b = a * a
        b.backward()
        assert a.grad == pytest.approx(6.0)

    def test_complex_reuse(self):
        """f = d * e where d = a*b, e = a+b  — a is used in two paths."""
        a = Value(-2.0)
        b = Value(3.0)
        d = a * b
        e = a + b
        f = d * e
        f.backward()
        # f = (a*b)*(a+b) = a^2*b + a*b^2
        # df/da = 2ab + b^2 = 2*(-2)*3 + 9 = -3
        assert a.grad == pytest.approx(-3.0)
        # df/db = a^2 + 2ab = 4 + (-12) = -8
        assert b.grad == pytest.approx(-8.0)


# -- Numerical gradient checking --------------------------------------------

class TestNumericalGradient:

    @staticmethod
    def numerical_grad(build_fn, var_index, h=1e-5):
        """Compute the numerical gradient for variable at var_index."""
        # Forward at current point
        variables1, out1 = build_fn()
        val1 = out1.data

        # Forward with nudge
        variables2, out2 = build_fn()
        variables2[var_index].data += h
        # re-evaluate: need to rebuild
        variables3, out3 = build_fn()
        variables3[var_index].data += h
        # Since we can't re-evaluate in-place easily, use the simple method
        # from the lecture: evaluate twice
        return (out3.data - out1.data) / h  # This won't work for Value

    def test_gradient_vs_numerical(self):
        """Verify analytical gradients match numerical gradients."""
        h = 1e-5

        def forward(a_val, b_val, c_val):
            a = Value(a_val)
            b = Value(b_val)
            c = Value(c_val)
            # f = (a + b) * c
            out = (a + b) * c
            return a, b, c, out

        a0, b0, c0 = 2.0, -3.0, 4.0

        # Analytical gradients
        a, b, c, out = forward(a0, b0, c0)
        out.backward()
        grad_a_analytical = a.grad
        grad_b_analytical = b.grad
        grad_c_analytical = c.grad

        # Numerical gradients
        _, _, _, out1 = forward(a0 + h, b0, c0)
        _, _, _, out0a = forward(a0, b0, c0)
        grad_a_numerical = (out1.data - out0a.data) / h

        _, _, _, out2 = forward(a0, b0 + h, c0)
        _, _, _, out0b = forward(a0, b0, c0)
        grad_b_numerical = (out2.data - out0b.data) / h

        _, _, _, out3 = forward(a0, b0, c0 + h)
        _, _, _, out0c = forward(a0, b0, c0)
        grad_c_numerical = (out3.data - out0c.data) / h

        assert grad_a_analytical == pytest.approx(grad_a_numerical, abs=1e-4)
        assert grad_b_analytical == pytest.approx(grad_b_numerical, abs=1e-4)
        assert grad_c_analytical == pytest.approx(grad_c_numerical, abs=1e-4)

    def test_tanh_gradient_vs_numerical(self):
        """Verify tanh gradient numerically."""
        h = 1e-5
        x_val = 0.8813735870195432

        x = Value(x_val)
        o = x.tanh()
        o.backward()
        grad_analytical = x.grad

        o1 = Value(x_val + h).tanh()
        o0 = Value(x_val).tanh()
        grad_numerical = (o1.data - o0.data) / h

        assert grad_analytical == pytest.approx(grad_numerical, abs=1e-4)


# -- Cross-check against PyTorch --------------------------------------------

class TestPyTorchCrossCheck:

    def test_neuron_matches_pytorch(self):
        """Verify micrograd matches PyTorch for a single neuron computation."""
        torch = pytest.importorskip("torch")

        # micrograd
        x1 = Value(2.0)
        x2 = Value(0.0)
        w1 = Value(-3.0)
        w2 = Value(1.0)
        b = Value(6.8813735870195432)
        n = x1 * w1 + x2 * w2 + b
        o = n.tanh()
        o.backward()

        # PyTorch
        tx1 = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        tx2 = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        tw1 = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
        tw2 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        tb = torch.tensor([6.8813735870195432], dtype=torch.float64, requires_grad=True)
        tn = tx1 * tw1 + tx2 * tw2 + tb
        to = torch.tanh(tn)
        to.backward()

        # Compare forward
        assert o.data == pytest.approx(to.data.item(), abs=1e-6)

        # Compare gradients
        assert x1.grad == pytest.approx(tx1.grad.item(), abs=1e-4)
        assert w1.grad == pytest.approx(tw1.grad.item(), abs=1e-4)
        assert x2.grad == pytest.approx(tx2.grad.item(), abs=1e-4)
        assert w2.grad == pytest.approx(tw2.grad.item(), abs=1e-4)
        assert b.grad == pytest.approx(tb.grad.item(), abs=1e-4)

    def test_expression_matches_pytorch(self):
        """Verify a more complex expression against PyTorch."""
        torch = pytest.importorskip("torch")

        # micrograd
        a = Value(-4.0)
        b = Value(2.0)
        c = a + b
        d = a * b + b ** 3
        c = c + c + 1
        c = c + 1 + c + (-a)
        d = d + d * 2 + (b + a).tanh()
        d = d + 3 * d + (b - a).tanh()
        e = c - d
        f = e ** 2
        g = f / 2.0
        g = g + 10.0 / f
        g.backward()

        # PyTorch
        ta = torch.tensor([-4.0], dtype=torch.float64, requires_grad=True)
        tb = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        tc = ta + tb
        td = ta * tb + tb ** 3
        tc = tc + tc + 1
        tc = tc + 1 + tc + (-ta)
        td = td + td * 2 + (tb + ta).tanh()
        td = td + 3 * td + (tb - ta).tanh()
        te = tc - td
        tf = te ** 2
        tg = tf / 2.0
        tg = tg + 10.0 / tf
        tg.backward()

        assert g.data == pytest.approx(tg.data.item(), abs=1e-6)
        assert a.grad == pytest.approx(ta.grad.item(), abs=1e-6)
        assert b.grad == pytest.approx(tb.grad.item(), abs=1e-6)


# -- MLP training convergence -----------------------------------------------

class TestMLP:

    def test_mlp_parameter_count(self):
        random.seed(42)
        n = MLP(3, [4, 4, 1])
        params = n.parameters()
        # Layer 0: 4 neurons * (3 weights + 1 bias) = 16
        # Layer 1: 4 neurons * (4 weights + 1 bias) = 20
        # Layer 2: 1 neuron  * (4 weights + 1 bias) = 5
        assert len(params) == 41

    def test_mlp_forward_produces_value(self):
        random.seed(42)
        n = MLP(3, [4, 4, 1])
        out = n([1.0, 2.0, 3.0])
        assert isinstance(out, Value)
        assert -1.0 <= out.data <= 1.0  # tanh output range

    def test_mlp_training_converges(self):
        """Train a small MLP and verify the loss decreases."""
        random.seed(42)
        n = MLP(3, [4, 4, 1])

        xs = [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ]
        ys = [1.0, -1.0, -1.0, 1.0]

        initial_loss = None
        for k in range(100):
            # forward
            ypred = [n(x) for x in xs]
            loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

            if k == 0:
                initial_loss = loss.data

            # backward
            for p in n.parameters():
                p.grad = 0.0
            loss.backward()

            # update
            for p in n.parameters():
                p.data += -0.05 * p.grad

        final_loss = loss.data
        assert final_loss < initial_loss, "Loss should decrease during training"
        assert final_loss < 0.5, f"Final loss {final_loss} is too high"

    def test_mlp_predictions_after_training(self):
        """After training, predictions should be close to targets."""
        random.seed(42)
        n = MLP(3, [4, 4, 1])

        xs = [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ]
        ys = [1.0, -1.0, -1.0, 1.0]

        for _ in range(200):
            ypred = [n(x) for x in xs]
            loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
            for p in n.parameters():
                p.grad = 0.0
            loss.backward()
            for p in n.parameters():
                p.data += -0.05 * p.grad

        ypred = [n(x) for x in xs]
        for pred, target in zip(ypred, ys):
            assert pred.data == pytest.approx(target, abs=0.3), (
                f"Prediction {pred.data} not close enough to target {target}"
            )
