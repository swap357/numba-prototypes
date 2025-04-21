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

from egglog import (
    Bool,
    String,
    Unit,
    Vec,
    function,
    i64,
    i64Like,
    rule,
    ruleset,
    set_,
    union,
)
from llvmlite import ir
from sealir import ase
from sealir.eqsat.py_eqsat import Py_NotIO
from sealir.eqsat.rvsdg_eqsat import (
    PortList,
    Region,
    Term,
    TermList,
    i64,
)
from sealir.rvsdg import grammar as rg

from ch03_egraph_program_rewrites import (
    run_test,
)
from ch04_1_typeinfer_controlflow import Backend as Ch04_1_Backend
from ch04_1_typeinfer_controlflow import (
    ExtendEGraphToRVSDG as _ch04_1_ExtendEGraphToRVSDG,
)
from ch04_1_typeinfer_controlflow import (
    Grammar,
    Int64,
    MyCostModel,
    NbOp_Base,
    SExpr,
    Type,
    TypeBool,
    TypedIns,
    TypeInt64,
    TypeVar,
    _wc,
    base_ruleset,
    compiler_pipeline,
    facts_function_types,
    ruleset_type_infer_failure_report,
    ruleset_type_infer_float,
)

# Define type inference for loop regions:


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
    ty: Type,
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
        ty == TypeVar(ports.getValue(start)).getType(),
    ).then(
        set_(TypeVar(operands[start - 1]).getType()).to(ty),
        union(TypeVar(ports.getValue(start))).with_(
            TypeVar(loop.getPort(start - 1))
        ),
    )


# Define rulesets for extra operations needed:


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
    ).then(union(x).with_(Term.LiteralBool(False)))
    yield rule(
        y == Py_NotIO(io=io, term=x),
        TypeVar(x).getType() == TypeInt64,
    ).then(
        union(y.getPort(0)).with_(io),
        union(y.getPort(1)).with_(Nb_Not_Int64(x)),
        set_(TypeVar(Nb_Not_Int64(x)).getType()).to(TypeBool),
    )


class NbOp_Not_Int64(NbOp_Base):
    operand: SExpr


class NbOp_Undef_Int64(NbOp_Base): ...


# Extend EGraphToRVSDG conversion from Ch4.1 to handle the extra operations


class ExtendEGraphToRVSDG(_ch04_1_ExtendEGraphToRVSDG):

    def handle_Term(self, op: str, children: dict | list, grm: Grammar):
        match op, children:
            case "Nb_Not_Int64", {"operand": operand}:
                return grm.write(NbOp_Not_Int64(operand=operand))
            case "Nb_Undef_Int64", {}:
                return grm.write(NbOp_Undef_Int64())
        return super().handle_Term(op, children, grm)


# Extend the LLVM Backend from Ch4.1


class Backend(Ch04_1_Backend):

    def lower_expr(self, expr, state):
        builder = state.builder
        match expr:

            case rg.Loop(body=rg.RegionEnd() as body, operands=operands):
                # process operands
                ops = []
                for op in operands:
                    ops.append((yield op))

                # Note this is a tail loop.
                begin = body.begin

                with state.push(*ops):
                    loopentry_values = yield begin

                bb_before = builder.basic_block
                bb_loopbody = builder.append_basic_block("loopbody")
                bb_endloop = builder.append_basic_block("endloop")
                builder.branch(bb_loopbody)
                # loop body
                builder.position_at_end(bb_loopbody)
                # setup phi nodes for loopback variables

                phis = []
                for i, var in enumerate(loopentry_values):
                    phi = builder.phi(var.type, name=f"loop_{i}")
                    phi.add_incoming(var, bb_before)
                    phis.append(phi)

                # generate body
                loop_memo = {begin: tuple(phis)}
                memo = ase.traverse(
                    body,
                    self.lower_expr,
                    state=state,
                    init_memo=loop_memo,
                )

                loopout_values = list(memo[body])
                # get loop condition
                loopcond = loopout_values.pop(0)

                # fix up phis
                for i, phi in enumerate(phis):
                    assert phi.type == loopout_values[i].type, (
                        phi.type,
                        loopout_values[i].type,
                    )
                    phi.add_incoming(loopout_values[i], builder.basic_block)
                # back jump
                builder.cbranch(loopcond, bb_loopbody, bb_endloop)
                # end loop
                builder.position_at_end(bb_endloop)
                # Returns the value from the loop body because this is a tail loop
                return loopout_values

            case NbOp_Not_Int64(operand):
                opval = yield operand
                return builder.icmp_unsigned("==", opval, opval.type(0))

            case NbOp_Undef_Int64():
                return ir.IntType(64)(ir.Undefined)

        return (yield from super().lower_expr(expr, state))


def example(init, n):
    c = float(init)
    i = 0
    while i < n:
        c = c + float(i)
        i = i + 1
    return c


if __name__ == "__main__":
    jt = compiler_pipeline(
        example,
        argtypes=(Int64, Int64),
        ruleset=(
            base_ruleset
            | facts_function_types
            | ruleset_others
            | ruleset_propagate_typeof_loops
            | ruleset_type_infer_float
            | ruleset_type_infer_failure_report
        ),
        verbose=True,
        converter_class=ExtendEGraphToRVSDG,
        cost_model=MyCostModel(),
        backend=Backend(),
    )
    run_test(example, jt, (10, 7), verbose=True)
