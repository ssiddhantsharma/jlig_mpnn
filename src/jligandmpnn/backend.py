"""from_torch machinery + basic layers (Linear, LayerNorm, Embedding)."""

from dataclasses import fields
from functools import singledispatch

import equinox as eqx
import jax
import torch
from jax import numpy as jnp


@singledispatch
def from_torch(x):
    raise NotImplementedError(f"from_torch not implemented for {type(x)}: {x}")


from_torch.register(torch.Tensor, lambda x: jnp.asarray(x.detach().cpu().numpy()))
from_torch.register(int, lambda x: x)
from_torch.register(float, lambda x: x)
from_torch.register(bool, lambda x: x)
from_torch.register(type(None), lambda x: x)
from_torch.register(tuple, lambda x: tuple(map(from_torch, x)))
from_torch.register(list, lambda x: [from_torch(v) for v in x])
from_torch.register(dict, lambda x: {k: from_torch(v) for k, v in x.items()})
from_torch.register(torch.nn.ModuleList, lambda x: [from_torch(m) for m in x])
from_torch.register(torch.nn.GELU, lambda _: lambda x: jax.nn.gelu(x, approximate=False))
from_torch.register(torch.nn.Dropout, lambda _: lambda x: x)  # inference: identity


def register_from_torch(torch_type):
    def decorator(cls):
        from_torch.register(torch_type, cls.from_torch)
        return cls

    return decorator


class AbstractFromTorch(eqx.Module):
    @classmethod
    def from_torch(cls, model: torch.nn.Module):
        field_types = {f.name: f.type for f in fields(cls)}
        kwargs = {c: from_torch(m) for c, m in model.named_children()} | {
            n: from_torch(p) for n, p in model.named_parameters(recurse=False)
        }
        for name, typ in field_types.items():
            if not hasattr(model, name):
                if not isinstance(None, typ):
                    raise ValueError(f"missing non-optional field {name} for {cls}")
                kwargs[name] = None
            else:
                kwargs[name] = from_torch(getattr(model, name))
        extra = kwargs.keys() - field_types.keys()
        if extra:
            raise ValueError(f"torch props not in {cls}: {extra}")
        return cls(**kwargs)


@register_from_torch(torch.nn.Linear)
class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array | None

    def __call__(self, x):
        o = x @ self.weight.T
        return o if self.bias is None else o + self.bias

    @staticmethod
    def from_torch(m: torch.nn.Linear):
        return Linear(weight=from_torch(m.weight), bias=from_torch(m.bias))


@register_from_torch(torch.nn.LayerNorm)
class LayerNorm(eqx.Module):
    weight: jax.Array | None
    bias: jax.Array | None
    eps: float

    def __call__(self, x):
        mean = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)  # biased, matches torch
        x = (x - mean) / jnp.sqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x

    @staticmethod
    def from_torch(m: torch.nn.LayerNorm):
        return LayerNorm(weight=from_torch(m.weight), bias=from_torch(m.bias), eps=m.eps)


@register_from_torch(torch.nn.Embedding)
class Embedding(eqx.Module):
    weight: jax.Array

    def __call__(self, idx):
        return self.weight[idx]

    @staticmethod
    def from_torch(m: torch.nn.Embedding):
        return Embedding(weight=from_torch(m.weight))
