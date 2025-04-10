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

# ## Ch 4 Part 1. Fully typing a scalar function.
#
# We will consider control-flow constructs. We went from
# per-operation inference, to considering the entire function.


from __future__ import annotations

from egglog import (
    EGraph,
    Expr,
    String,
    StringLike,
    Vec,
    delete,
    function,
    i64,
    i64Like,
    rewrite,
    rule,
    ruleset,
    set_,
    union,
)
from sealir import grammar, rvsdg
from sealir.eqsat import rvsdg_eqsat
from sealir.eqsat.py_eqsat import Py_AddIO, Py_GtIO, Py_SubIO
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import (
    GraphRoot,
    InPorts,
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
    egraph_extraction,
)
from sealir.llvm_pyapi_backend import SSAValue
from sealir.rvsdg import grammar as rg

from ch03_egraph_program_rewrites import (
    backend,
    frontend,
    jit_compile,
    ruleset_const_propagate,
    run_test,
)
from ch04_0_typeinfer_scalar import (
    basic_ruleset,
    middle_end,
)
from utils import IN_NOTEBOOK


def compiler_pipeline(
    fn,
    *,
    verbose=False,
    ruleset,
    converter_class=EGraphToRVSDG,
    codegen_extension=None,
    cost_model=None,
):
    rvsdg_expr, dbginfo = frontend(fn)

    # Middle end
    def define_egraph(egraph: EGraph, func):
        root = GraphRoot(func)
        egraph.let("root", root)

        egraph.run(ruleset.saturate())
        if verbose and IN_NOTEBOOK:
            # For inspecting the egraph
            egraph.display(graphviz=True)
        # egraph.display()

    cost, extracted = middle_end(
        rvsdg_expr,
        define_egraph,
        converter_class=converter_class,
        cost_model=cost_model,
    )
    print("Extracted from EGraph".center(80, "="))
    print("cost =", cost)
    print(rvsdg.format_rvsdg(extracted))

    # llmod = backend(extracted, codegen_extension=codegen_extension)
    # if verbose:
    #     print("LLVM module".center(80, "="))
    #     print(llmod)
    # return jit_compile(llmod, extracted)


_wc = wildcard


class Type(Expr):
    def __init__(self, name: StringLike): ...

    def __or__(self, other: Type) -> Type: ...


@function
def TypeOf(x: Term) -> Type: ...


@function
def Nb_Gt_Int64(lhs: Term, rhs: Term) -> Term: ...


@function
def Nb_Add_Int64(lhs: Term, rhs: Term) -> Term: ...


@function
def Nb_Sub_Int64(lhs: Term, rhs: Term) -> Term: ...


SExpr = rvsdg.grammar.SExpr


class NbOp_Base(grammar.Rule):
    pass


class NbOp_Type(NbOp_Base):
    name: str


class NbOp_InTypeAttr(NbOp_Base):
    idx: int
    type: NbOp_Type


class NbOp_OutTypeAttr(NbOp_Base):
    idx: int
    type: NbOp_Type


class NbOp_Gt_Int64(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Add_Int64(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Sub_Int64(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class Grammar(grammar.Grammar):
    start = rvsdg.Grammar.start | NbOp_Base


class ExtendEGraphToRVSDG(EGraphToRVSDG):
    grammar = Grammar

    def handle_region_attributes(self, key: str, grm: Grammar):
        attrs = []
        typedargs = list(self.search_calls(key, "TypedIns"))
        if typedargs:
            [typedarg] = typedargs
            for key_arg in self.search_method_calls(typedarg, "arg"):
                _k_self, k_idx = self.get_children(key_arg)
                idx = self.dispatch(k_idx, grm)

                typs = list(
                    self.filter_by_type(
                        "Type", self.search_eclass_siblings(key_arg)
                    )
                )
                if len(typs) == 1:
                    typ = self.dispatch(typs[0], grm)
                    attrs.append(grm.write(NbOp_InTypeAttr(idx=idx, type=typ)))
                else:
                    assert len(typs) == 0, "multiple types"

        typedouts = list(self.search_calls(key, "TypedOuts"))
        if typedouts:
            [typedout] = typedouts
            for key_at in self.search_method_calls(typedout, "at"):
                _k_self, k_idx = self.get_children(key_at)
                idx = self.dispatch(k_idx, grm)

                typs = list(
                    self.filter_by_type(
                        "Type", self.search_eclass_siblings(key_at)
                    )
                )
                if len(typs) == 1:
                    typ = self.dispatch(typs[0], grm)
                    attrs.append(
                        grm.write(NbOp_OutTypeAttr(idx=idx, type=typ))
                    )
                else:
                    assert len(typs) == 0, "multiple types"

        return grm.write(rg.Attrs(tuple(attrs)))

    def handle_Type(
        self, key: str, op: str, children: dict | list, grm: Grammar
    ):
        assert op == "Type"
        [name] = children
        return grm.write(NbOp_Type(name))

    def handle_Term(self, op: str, children: dict | list, grm: Grammar):
        match op, children:

            case "Nb_Gt_Int64", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Gt_Int64(lhs=lhs, rhs=rhs))
            case "Nb_Add_Int64", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Add_Int64(lhs=lhs, rhs=rhs))
            case "Nb_Sub_Int64", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Sub_Int64(lhs=lhs, rhs=rhs))
            case _:
                # Use parent's implementation for other terms.
                return super().handle_Term(op, children, grm)


# The LLVM code-generation also needs an extension:


def codegen_extension(expr, args, builder, pyapi):
    match expr._head, args:
        case "NbOp_Add_Int64", (lhs, rhs):
            return SSAValue(builder.add(lhs.value, rhs.value))
        case "NbOp_Sub_Int64", (lhs, rhs):
            return SSAValue(builder.sub(lhs.value, rhs.value))
    return NotImplemented


class MyCostModel(CostModel):
    def get_cost_function(self, nodename, op, cost, nodes, child_costs):
        self_cost = None
        match op:
            case "Nb_Add_Int64" | "Nb_Sub_Int64" | "Nb_Gt_Int64":
                self_cost = 0.1

        if self_cost is not None:
            return self_cost + sum(child_costs)

        # Fallthrough to parent's cost function
        return super().get_cost_function(
            nodename, op, cost, nodes, child_costs
        )


TypeInt64 = Type("Int64")
TypeBool = Type("Bool")


@ruleset
def ruleset_type_infer_gt(io: Term, x: Term, y: Term, add: Term):
    yield rule(
        add == Py_GtIO(io, x, y),
        TypeOf(x) == TypeInt64,
        TypeOf(y) == TypeInt64,
    ).then(
        # convert to a typed operation
        union(add.getPort(1)).with_(Nb_Gt_Int64(x, y)),
        # shortcut io
        union(add.getPort(0)).with_(io),
        # output type
        union(TypeOf(add.getPort(1))).with_(TypeBool),
    )


@ruleset
def ruleset_type_infer_add(io: Term, x: Term, y: Term, add: Term):
    yield rule(
        add == Py_AddIO(io, x, y),
        TypeOf(x) == TypeInt64,
        TypeOf(y) == TypeInt64,
    ).then(
        # convert to a typed operation
        union(add.getPort(1)).with_(Nb_Add_Int64(x, y)),
        # shortcut io
        union(add.getPort(0)).with_(io),
        # output type
        union(TypeOf(add.getPort(1))).with_(TypeInt64),
    )


@ruleset
def ruleset_type_infer_sub(io: Term, x: Term, y: Term, add: Term):
    yield rule(
        add == Py_SubIO(io, x, y),
        TypeOf(x) == TypeInt64,
        TypeOf(y) == TypeInt64,
    ).then(
        # convert to a typed operation
        union(add.getPort(1)).with_(Nb_Sub_Int64(x, y)),
        # shortcut io
        union(add.getPort(0)).with_(io),
        # output type
        union(TypeOf(add.getPort(1))).with_(TypeInt64),
    )


@ruleset
def ruleset_type_unify(ta: Type, tb: Type):
    yield rewrite(ta | tb, subsume=True).to(ta, ta == tb)


@ruleset
def ruleset_propagate_typeof_ifelse(
    then_region: Region,
    else_region: Region,
    idx: i64,
    src: Term,
    ifelse: Term,
    then_ports: PortList,
    else_ports: PortList,
    operands: Vec[Term],
):

    yield rule(
        # Propagate operand types into the contained regions
        Term.IfElse(
            cond=_wc(Term),
            then=Term.RegionEnd(region=then_region, ports=_wc(PortList)),
            orelse=Term.RegionEnd(region=else_region, ports=_wc(PortList)),
            operands=TermList(operands),
        ),
        then_region.get(idx),
    ).then(
        union(TypeOf(operands[idx])).with_(TypedIns(then_region).arg(idx)),
        union(TypeOf(operands[idx])).with_(TypedIns(else_region).arg(idx)),
    )

    yield rule(
        # Propagate output types from the contained regions
        ifelse
        == Term.IfElse(
            cond=_wc(Term),
            then=Term.RegionEnd(region=_wc(Region), ports=then_ports),
            orelse=Term.RegionEnd(region=_wc(Region), ports=else_ports),
            operands=TermList(operands),
        ),
        ta := TypeOf(then_ports.getValue(idx)),
        tb := TypeOf(else_ports.getValue(idx)),
    ).then(
        union(ta | tb).with_(TypeOf(ifelse.getPort(idx))),
    )


class TypedIns(Expr):
    def __init__(self, region: Region): ...

    def arg(self, idx: i64Like) -> Type: ...


class TypedOuts(Expr):
    def __init__(self, region: Region): ...

    def at(self, idx: i64Like) -> Type: ...


@ruleset
def ruleset_region_args(
    region: Region,
    attrs: TypedIns,
    idx: i64,
    typ: Type,
    term: Term,
    portlist: PortList,
):
    yield rule(
        # Inputs
        typ == TypedIns(region).arg(idx),
        term == region.get(idx),
    ).then(
        union(TypeOf(term)).with_(typ),
    )

    yield rule(
        # Outputs
        term == Term.RegionEnd(region=region, ports=portlist),
        typ == TypeOf(portlist.getValue(idx)),
    ).then(
        union(TypedOuts(region).at(idx)).with_(typ),
    )


@ruleset
def facts_argument_types(
    outports: PortList,
    func_uid: String,
    reg_uid: String,
    fname: String,
    region: Region,
    arg_x: Term,
    arg_y: Term,
):
    if False:
        return
    yield rule(
        GraphRoot(
            Term.Func(
                body=Term.RegionEnd(region=region, ports=outports),
                uid=func_uid,
                fname=fname,
            )
        ),
        region == Region(uid=reg_uid, inports=_wc(InPorts)),
    ).then(
        union(TypedIns(region).arg(1)).with_(TypeInt64),
        union(TypedIns(region).arg(2)).with_(TypeInt64),
    )


def example(a, b):
    if a > b:
        z = a - b
    else:
        z = b - a
    return z - a


if __name__ == "__main__":
    jt = compiler_pipeline(
        example,
        ruleset=(
            basic_ruleset
            | ruleset_propagate_typeof_ifelse
            | ruleset_type_unify
            | ruleset_type_infer_gt
            | ruleset_type_infer_add
            | ruleset_type_infer_sub
            | ruleset_region_args
            | facts_argument_types
        ),
        verbose=True,
        converter_class=ExtendEGraphToRVSDG,
        codegen_extension=codegen_extension,
        cost_model=MyCostModel(),
    )
    res = jt(10, 7)
    run_test(example, jt, (10, 7), verbose=True)
