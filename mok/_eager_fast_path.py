"""Eager fast path for MoK custom ops.

MoK registers each op with ``torch.library.custom_op`` so that ``torch.compile`` sees it as one
opaque graph node with a schema, a fake implementation and mutation annotations. In eager mode
every call still goes through the dispatcher: ``OpOverload.__call__`` enters C++, the Autograd
kernel calls back into Python and redispatches, the ADInplaceOrView kernel fills in defaults and
bumps the version counters of mutated arguments, and the backend kernel calls the op body and
checks output aliasing. MoK calls its ops with plain tensors and nothing for autograd to record,
so these steps do no work, but on torch 2.13 they cost about 40-200 us of host time per call.

``eager_fast_path_custom_op`` registers the op exactly like ``torch.library.custom_op`` and
returns the same ``CustomOpDef`` object with its class switched to ``EagerFastPathOp``, so every
``torch.library`` API keeps accepting it. Calling it runs the op body directly when that is
equivalent to the dispatcher path, and dispatches otherwise. The direct call is taken only when
all of these hold:

* no ``torch.compile`` tracing, JIT tracing or functorch transform is active,
* no ``TorchDispatchMode`` (for example ``FakeTensorMode``) and no ``TorchFunctionMode`` is active,
* the op still has only the kernel it was defined with: no other backend kernel, autocast or vmap
  rule has been registered and no kernel has been disabled,
* every tensor argument is a plain ``torch.Tensor`` or ``torch.nn.Parameter`` that is not on the
  meta device, and autograd would not record the call: grad mode is off, or no tensor argument
  requires grad,
* every other argument already has the exact Python type of its schema type, so that the
  dispatcher's argument conversion would be a no-op (list elements are not checked).

The direct call bumps the version counters of mutated arguments, as the ADInplaceOrView kernel
does, and runs the body below the Autograd dispatch key, as the dispatcher does. While a profiler
is active it records the same ``<namespace>::<op>`` event the dispatcher would. It skips only the
output aliasing check.

Environment variables, read at import time:

* ``MOK_EAGER_FAST_PATH=0`` always uses the dispatcher.
* ``MOK_EAGER_FAST_PATH_STATS=1`` counts direct and dispatcher calls per op in eager mode,
  together with the first condition that ruled out the direct call, and prints the counts at exit.
"""

from __future__ import annotations

import atexit
import collections
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
import torch.autograd.profiler as _autograd_profiler
from torch._library.custom_ops import CustomOpDef

ENABLED: bool = os.environ.get("MOK_EAGER_FAST_PATH", "1") != "0"
_STATS_ENABLED: bool = os.environ.get("MOK_EAGER_FAST_PATH_STATS", "0") == "1"
_STATS: collections.Counter[tuple[str, str]] = collections.Counter()

_TENSOR = "tensor"


def _python_type_check(arg_type: torch._C.Type) -> tuple[str | type | tuple[type, ...], bool]:
    """(kind, optional) for a schema argument type; kind is _TENSOR, a Python type or ``object``."""
    optional = False
    if isinstance(arg_type, torch._C.OptionalType):
        optional = True
        arg_type = arg_type.getElementType()
    if isinstance(arg_type, torch._C.TensorType):
        return _TENSOR, optional
    if isinstance(arg_type, torch._C.ListType):
        if isinstance(arg_type.getElementType(), torch._C.TensorType):
            return _TENSOR, optional
        return list, optional
    for schema_type, python_type in (
        (torch._C.BoolType, bool),
        (torch._C.IntType, int),
        (torch._C.FloatType, float),
        (torch._C.StringType, str),
    ):
        if isinstance(arg_type, schema_type):
            return python_type, optional
    # Anything else (dtype, device, Scalar, ...) is converted by the dispatcher: never bypass it.
    return object, optional


def _is_plain_tensor(value: Any, check_grad: bool) -> bool:
    if isinstance(value, (list, tuple)):
        return all(_is_plain_tensor(item, check_grad) for item in value)
    value_type = type(value)
    if value_type is not torch.Tensor and value_type is not torch.nn.Parameter:
        # Tensor subclasses need their __torch_dispatch__; anything else should get the
        # dispatcher's argument error.
        return False
    if value.is_meta:
        return False
    return not (check_grad and value.requires_grad)


class EagerFastPathOp(CustomOpDef):
    """A ``CustomOpDef`` that calls its body directly in plain eager mode.

    Instances are made by ``eager_fast_path_custom_op``, which switches the class of the
    ``CustomOpDef`` returned by ``torch.library.custom_op``; the dispatcher registration is
    untouched.
    """

    def _mok_init(self, impl: Callable[..., Any], *, allow_fast_path: bool) -> None:
        self._mok_impl = impl
        self._mok_allowed = allow_fast_path
        arguments = list(self._opoverload._schema.arguments)
        checks = []
        for index, argument in enumerate(arguments):
            kind, optional = _python_type_check(argument.type)
            checks.append((index, argument.name, kind, optional))
        self._mok_checks = tuple(checks)
        self._mok_mutated = tuple(
            (index, argument.name)
            for index, argument in enumerate(arguments)
            if argument.alias_info is not None and argument.alias_info.is_write
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Checked first so that Dynamo only ever sees the plain dispatcher call.
        if torch.compiler.is_compiling():
            return self._opoverload(*args, **kwargs)
        if _STATS_ENABLED:
            reason = self._mok_fallback_reason(args, kwargs)
            _STATS[(self._qualname, reason or "direct")] += 1
            if reason is None:
                return self._mok_call_direct(args, kwargs)
            return self._opoverload(*args, **kwargs)
        if ENABLED and self._mok_fallback_reason(args, kwargs) is None:
            return self._mok_call_direct(args, kwargs)
        return self._opoverload(*args, **kwargs)

    def _mok_call_direct(self, args: Sequence[Any], kwargs: dict[str, Any]) -> Any:
        num_args = len(args)
        for index, name in self._mok_mutated:
            value = args[index] if index < num_args else kwargs.get(name)
            if value is not None:
                torch.autograd.graph.increment_version(value)
        # The dispatcher runs the body below the Autograd key (custom_op's forward_no_grad);
        # keep the same thread-local state for the torch ops the body calls.
        with torch._C._AutoDispatchBelowAutograd():
            if _autograd_profiler._is_profiler_enabled:
                with torch._C._profiler._RecordFunctionFast(self._qualname):
                    return self._mok_impl(*args, **kwargs)
            return self._mok_impl(*args, **kwargs)

    def _mok_fallback_reason(self, args: Sequence[Any], kwargs: dict[str, Any]) -> str | None:
        """None if the direct call is equivalent to dispatching, else why it is not."""
        if not ENABLED:
            return "disabled"
        if not self._mok_allowed:
            return "device-specific registration"
        if (
            torch._C._len_torch_dispatch_stack()
            or torch._C._is_torch_function_mode_enabled()
            or torch._C._are_functorch_transforms_active()
            or torch._C._get_tracing_state() is not None
        ):
            return "mode, transform or tracer active"
        if (
            len(self._backend_fns) != 1
            or None not in self._backend_fns
            or self._disabled_kernel
            or self._autocast_dtype
            or self._vmap_fn is not None
        ):
            return "other kernel or rule registered"
        check_grad = torch.is_grad_enabled()
        num_args = len(args)
        for index, name, kind, optional in self._mok_checks:
            if index < num_args:
                value = args[index]
            elif name in kwargs:
                value = kwargs[name]
            else:
                continue  # default value, already of the right type
            if value is None:
                if optional:
                    continue
                return "argument type"
            if kind is _TENSOR:
                if not _is_plain_tensor(value, check_grad):
                    return "tensor subclass, meta tensor or requires grad"
            elif type(value) is not kind:
                return "argument type"
        return None


def eager_fast_path_custom_op(
    name: str,
    fn: Callable[..., Any] | None = None,
    /,
    *,
    mutates_args: str | Iterable[str],
    device_types: str | Sequence[str] | None = None,
    schema: str | None = None,
) -> Any:
    """``torch.library.custom_op`` that returns an ``EagerFastPathOp``."""

    def inner(impl: Callable[..., Any]) -> EagerFastPathOp:
        op = torch.library.custom_op(
            name, mutates_args=mutates_args, device_types=device_types, schema=schema
        )(impl)
        op.__class__ = EagerFastPathOp
        op._mok_init(impl, allow_fast_path=device_types is None)
        return op

    if fn is None:
        return inner
    return inner(fn)


def stats() -> dict[tuple[str, str], int]:
    """Call counts per (op, path) since import; populated only with MOK_EAGER_FAST_PATH_STATS=1."""
    return dict(_STATS)


def _print_stats() -> None:
    if not _STATS:
        return
    rank = os.environ.get("RANK", "0")
    lines = [f"[mok rank {rank}] eager fast path calls (op, path): count"]
    lines += [f"  {op} {path}: {count}" for (op, path), count in sorted(_STATS.items())]
    print("\n".join(lines), file=sys.stderr, flush=True)


if _STATS_ENABLED:
    atexit.register(_print_stats)
