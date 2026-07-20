"""
MicroGPT core (Karpathy): Value autograd + tiny GPT, adapted as a library.
Original algorithm preserved; wired for train/infer of Grafix triples.
"""

from __future__ import annotations

import math
import random
from typing import Any


class Value:
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


def matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


class MicroGPT:
    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 32,
        n_head: int = 4,
        n_layer: int = 1,
        block_size: int = 96,
        seed: int = 42,
    ):
        random.seed(seed)
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.head_dim = n_embd // n_head

        self.state_dict: dict[str, Any] = {
            "wte": matrix(vocab_size, n_embd),
            "wpe": matrix(block_size, n_embd),
            "lm_head": matrix(vocab_size, n_embd),
        }
        for i in range(n_layer):
            self.state_dict[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)
            self.state_dict[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)
            self.state_dict[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)
            self.state_dict[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd)
            self.state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
            self.state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)

        self.params = [p for mat in self.state_dict.values() for row in mat for p in row]

    def gpt(self, token_id, pos_id, keys, values):
        tok_emb = self.state_dict["wte"][token_id]
        pos_emb = self.state_dict["wpe"][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm(x)

        for li in range(self.n_layer):
            x_residual = x
            x = rmsnorm(x)
            q = linear(x, self.state_dict[f"layer{li}.attn_wq"])
            k = linear(x, self.state_dict[f"layer{li}.attn_wk"])
            v = linear(x, self.state_dict[f"layer{li}.attn_wv"])
            keys[li].append(k)
            values[li].append(v)
            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs : hs + self.head_dim]
                k_h = [ki[hs : hs + self.head_dim] for ki in keys[li]]
                v_h = [vi[hs : hs + self.head_dim] for vi in values[li]]
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(self.head_dim)) / self.head_dim**0.5
                    for t in range(len(k_h))
                ]
                attn_weights = softmax(attn_logits)
                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(self.head_dim)
                ]
                x_attn.extend(head_out)
            x = linear(x_attn, self.state_dict[f"layer{li}.attn_wo"])
            x = [a + b for a, b in zip(x, x_residual)]

            x_residual = x
            x = rmsnorm(x)
            x = linear(x, self.state_dict[f"layer{li}.mlp_fc1"])
            x = [xi.relu() for xi in x]
            x = linear(x, self.state_dict[f"layer{li}.mlp_fc2"])
            x = [a + b for a, b in zip(x, x_residual)]

        return linear(x, self.state_dict["lm_head"])

    def loss_on_tokens(self, tokens: list[int]) -> Value:
        n = min(self.block_size, len(tokens) - 1)
        if n <= 0:
            return Value(0.0)
        keys = [[] for _ in range(self.n_layer)]
        values = [[] for _ in range(self.n_layer)]
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            if token_id >= self.vocab_size or target_id >= self.vocab_size:
                continue
            logits = self.gpt(token_id, pos_id, keys, values)
            probs = softmax(logits)
            losses.append(-probs[target_id].log())
        if not losses:
            return Value(0.0)
        return (1 / len(losses)) * sum(losses)

    def generate_until_bos(
        self,
        prompt_ids: list[int],
        bos_id: int,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
    ) -> list[int]:
        keys = [[] for _ in range(self.n_layer)]
        values = [[] for _ in range(self.n_layer)]
        # Rebuild cache correctly: forward prompt left-to-right
        for pos_id in range(len(prompt_ids) - 1):
            if pos_id >= self.block_size:
                break
            self.gpt(prompt_ids[pos_id], pos_id, keys, values)

        generated = list(prompt_ids)
        for _ in range(max_new_tokens):
            pos_id = len(generated) - 1
            if pos_id >= self.block_size:
                break
            logits = self.gpt(generated[-1], pos_id, keys, values)
            probs = softmax([l / temperature for l in logits])
            next_id = random.choices(range(self.vocab_size), weights=[p.data for p in probs])[0]
            if next_id == bos_id:
                break
            generated.append(next_id)
        return generated

    def zero_grad(self):
        for p in self.params:
            p.grad = 0

    def adam_step(self, m, v, step: int, lr: float, beta1=0.85, beta2=0.99, eps=1e-8, num_steps=1):
        lr_t = lr * (1 - step / max(num_steps, 1))
        for i, p in enumerate(self.params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat**0.5 + eps)
            p.grad = 0

    def export_weights(self) -> dict:
        out = {
            "config": {
                "vocab_size": self.vocab_size,
                "n_embd": self.n_embd,
                "n_head": self.n_head,
                "n_layer": self.n_layer,
                "block_size": self.block_size,
            },
            "weights": {},
        }
        for name, mat in self.state_dict.items():
            out["weights"][name] = [[p.data for p in row] for row in mat]
        return out

    def load_weights(self, weights: dict) -> None:
        for name, mat in weights.items():
            for i, row in enumerate(mat):
                for j, val in enumerate(row):
                    self.state_dict[name][i][j].data = float(val)
