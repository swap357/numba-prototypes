# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: sealir_basic_compiler
#     language: python
#     name: python3
# ---

# ## Ch 4 Part 2. Fully typing a scalar function with loops
#
# We will consider simple loops.


from __future__ import annotations

import ctypes
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from egglog import (
    Bool,
    EGraph,
    Expr,
    String,
    StringLike,
    Unit,
    Vec,
    function,
    i64,
    i64Like,
    rewrite,
    rule,
    ruleset,
    set_,
    union,
)
from llvmlite import binding as llvm
from llvmlite import ir
from sealir import ase, grammar, rvsdg
from sealir.eqsat.py_eqsat import Py_AddIO, Py_GtIO, Py_NotIO, Py_SubIO
from sealir.eqsat.rvsdg_eqsat import (
    GraphRoot,
    InPorts,
    Port,
    PortList,
    Region,
    Term,
    TermList,
    i64,
    wildcard,
)
from sealir.eqsat.rvsdg_extract import (
    CostModel,
    EGraphToRVSDG,
)
from sealir.rvsdg import grammar as rg

from ch03_egraph_program_rewrites import (
    frontend,
    run_test,
)
from ch04_1_typeinfer_controlflow import (
    Backend,
    ExtendEGraphToRVSDG,
    Int64,
    MyCostModel,
    Type,
    TypeBool,
    TypedIns,
    TypeInt64,
    TypeVar,
    _wc,
    basic_ruleset,
    compiler_pipeline,
    facts_function_types,
    ruleset_propagate_typeof_ifelse,
    ruleset_region_types,
    ruleset_type_infer_add,
    ruleset_type_infer_gt,
    ruleset_type_infer_lt,
    ruleset_type_infer_sub,
    ruleset_type_unify,
)


@function
def assign_output_loop_typevar(
    start: i64Like,
    stop: i64Like,
    ports: PortList,
    operands: Vec[Term],
    loop: Term,
) -> Unit: ...


@ruleset
def ruleset_propagate_typeof_loops(
    loop: Term,
    body: Term,
    operands: Vec[Term],
    idx: i64,
    ports: PortList,
    region: Region,
    start: i64,
    stop: i64,
):
    yield rule(
        loop == Term.Loop(body=body, operands=TermList(operands)),
        body == Term.RegionEnd(region=region, ports=ports),
        region.get(idx),
    ).then(
        # propagate loop inputs
        union(TypeVar(operands[idx])).with_(TypedIns(region).arg(idx)),
    )

    yield rule(
        loop == Term.Loop(body=body, operands=TermList(operands)),
        body == Term.RegionEnd(region=region, ports=ports),
    ).then(
        # propagate loop outputs
        assign_output_loop_typevar(0, operands.length(), ports, operands, loop)
    )

    yield rule(
        assign_output_loop_typevar(start, stop, ports, operands, loop),
        start + 1 < stop,
    ).then(
        assign_output_loop_typevar(start + 1, stop, ports, operands, loop),
    )

    yield rule(
        assign_output_loop_typevar(start, stop, ports, operands, loop),
        start > 0,
    ).then(
        union(TypeVar(ports.getValue(start))).with_(
            TypeVar(operands[start - 1])
        ),
        union(TypeVar(ports.getValue(start))).with_(
            TypeVar(loop.getPort(start - 1))
        ),
    )


@function
def Nb_Not_Int64(operand: Term) -> Term: ...


@function
def Nb_Undef_Int64() -> Term: ...


@ruleset
def ruleset_others(x: Term, y: Term, io: Term):
    yield rule(x == Term.LiteralI64(_wc(i64))).then(
        set_(TypeVar(x).getType()).to(TypeInt64)
    )
    yield rule(x == Term.LiteralBool(_wc(Bool))).then(
        set_(TypeVar(x).getType()).to(TypeBool)
    )
    yield rule(x == Term.Undef(_wc(String))).then(
        set_(TypeVar(x).getType()).to(Type.simple("undef"))
    )
    yield rule(
        x == Term.Undef(_wc(String)),
        TypeVar(x).getType() == TypeInt64,
    ).then(union(x).with_(Nb_Undef_Int64()))
    yield rule(
        x == Term.Undef(_wc(String)),
        TypeVar(x).getType() == TypeBool,
    ).then(
        union(x).with_(Term.LiteralBool(False))
    )
    yield rule(
        y == Py_NotIO(io=io, term=x),
        TypeVar(x).getType() == TypeInt64,
    ).then(
        union(y.getPort(0)).with_(io),
        union(y.getPort(1)).with_(Nb_Not_Int64(x)),
        set_(TypeVar(Nb_Not_Int64(x)).getType()).to(TypeBool),
    )


def example(init, n):
    c = init
    i = 0
    while i < n:
        c = c + i
        i = i + 1
    return c


if __name__ == "__main__":
    jt = compiler_pipeline(
        example,
        argtypes=(Int64, Int64),
        ruleset=(
            basic_ruleset
            | ruleset_propagate_typeof_ifelse
            | ruleset_propagate_typeof_loops
            | ruleset_type_unify
            | ruleset_type_infer_gt
            | ruleset_type_infer_lt
            | ruleset_type_infer_add
            | ruleset_type_infer_sub
            | ruleset_region_types
            | facts_function_types
            | ruleset_others
        ),
        verbose=True,
        converter_class=ExtendEGraphToRVSDG,
        cost_model=MyCostModel(),
        backend=Backend(),
    )
    print(example(10, 7))
    run_test(example, jt, (10, 7), verbose=True)
