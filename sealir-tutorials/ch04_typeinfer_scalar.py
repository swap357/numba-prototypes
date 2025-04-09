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

# ## Ch 4. Type inference for scalar operations

from egglog import (
    EGraph,
    Expr,
    String,
    StringLike,
    function,
    rewrite,
    rule,
    ruleset,
    union,
)
from sealir import grammar, rvsdg
from sealir.eqsat import rvsdg_eqsat
from sealir.eqsat.py_eqsat import Py_AddIO
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import GraphRoot, PortList, Region, Term
from sealir.eqsat.rvsdg_extract import (
    CostModel,
    EGraphToRVSDG,
    egraph_extraction,
)
from sealir.llvm_pyapi_backend import SSAValue

from ch03_egraph_program_rewrites import (
    backend,
    frontend,
    jit_compile,
    ruleset_const_propagate,
    run_test,
)


def middle_end(rvsdg_expr, apply_to_egraph, converter_class, cost_model):
    # Convert to egraph
    memo = egraph_conversion(rvsdg_expr)

    func = memo[rvsdg_expr]

    egraph = EGraph()
    apply_to_egraph(egraph, func)

    # Extraction
    cost, extracted = egraph_extraction(
        egraph,
        rvsdg_expr,
        converter_class=converter_class,
        cost_model=cost_model,
    )
    return cost, extracted


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
        # For now, the middle end is just an identity function that exercise
        # the encoding into and out of egraph.
        root = GraphRoot(func)
        egraph.let("root", root)
        egraph.run(ruleset.saturate())
        if verbose:
            # For inspecting the egraph
            egraph.display(graphviz=True)

    cost, extracted = middle_end(
        rvsdg_expr,
        define_egraph,
        converter_class=converter_class,
        cost_model=cost_model,
    )
    print("Extracted from EGraph".center(80, "="))
    print("cost =", cost)
    print(rvsdg.format_rvsdg(extracted))

    llmod = backend(extracted, codegen_extension=codegen_extension)
    if verbose:
        print("LLVM module".center(80, "="))
        print(llmod)
    return jit_compile(llmod, extracted)


def add_x_y(x, y):
    return x + y


basic_ruleset = rvsdg_eqsat.ruleset_rvsdg_basic | ruleset_const_propagate


if __name__ == "__main__":
    # start with previous compiler pipeline
    jt = compiler_pipeline(add_x_y, ruleset=basic_ruleset, verbose=True)
    run_test(add_x_y, jt, (123, 321), verbose=True)


# adding type inference


class Type(Expr):
    def __init__(self, name: StringLike): ...


@function
def TypeOf(x: Term) -> Type: ...


@function
def Nb_Unbox_Int64(val: Term) -> Term: ...


@function
def Nb_Box_Int64(val: Term) -> Term: ...


@function
def Nb_Unboxed_Add_Int64(lhs: Term, rhs: Term) -> Term: ...


TypeInt64 = Type("Int64")


@ruleset
def ruleset_type_infer_add(io: Term, x: Term, y: Term, add: Term):
    yield rule(
        add == Py_AddIO(io, x, y),
        TypeOf(x) == TypeInt64,
        TypeOf(y) == TypeInt64,
    ).then(
        # convert to a typed operation
        union(add.getPort(1)).with_(
            Nb_Box_Int64(
                Nb_Unboxed_Add_Int64(Nb_Unbox_Int64(x), Nb_Unbox_Int64(y))
            )
        ),
        # shortcut io
        union(add.getPort(0)).with_(io),
        # output type
        union(TypeOf(add.getPort(1))).with_(TypeInt64),
    )


@ruleset
def facts_argument_types(
    outports: PortList,
    func_uid: String,
    fname: String,
    region: Region,
    arg_x: Term,
    arg_y: Term,
):
    yield rule(
        Term.Func(
            body=Term.RegionEnd(region=region, ports=outports),
            uid=func_uid,
            fname=fname,
        ),
        arg_x == region.get(1),
        arg_y == region.get(2),
    ).then(
        union(TypeOf(arg_x)).with_(TypeInt64),
        union(TypeOf(arg_y)).with_(TypeInt64),
    )


SExpr = rvsdg.grammar.SExpr


class _Root(grammar.Rule):
    pass


class NbOp_Unboxed_Add_Int64(_Root):
    lhs: SExpr
    rhs: SExpr


class NbOp_Unbox_Int64(_Root):
    val: SExpr


class NbOp_Box_Int64(_Root):
    val: SExpr


class Grammar(grammar.Grammar):
    start = rvsdg.Grammar.start | _Root


class ExtendEGraphToRVSDG(EGraphToRVSDG):
    grammar = Grammar

    def handle_Term(self, op: str, children: dict | list, grm: Grammar):
        match op, children:
            case "Nb_Unboxed_Add_Int64", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Unboxed_Add_Int64(lhs=lhs, rhs=rhs))
            case "Nb_Unbox_Int64", {"val": val}:
                return grm.write(NbOp_Unbox_Int64(val=val))
            case "Nb_Box_Int64", {"val": val}:
                return grm.write(NbOp_Box_Int64(val=val))
            case _:
                return super().handle_Term(op, children, grm)


def codegen_extension(expr, args, builder, pyapi):
    match expr._head, args:
        case "NbOp_Unboxed_Add_Int64", (lhs, rhs):
            return SSAValue(builder.add(lhs.value, rhs.value))
        case "NbOp_Unbox_Int64", (val,):
            return SSAValue(pyapi.long_as_longlong(val.value))
        case "NbOp_Box_Int64", (val,):
            return SSAValue(pyapi.long_from_longlong(val.value))
    return NotImplemented


class MyCostModel(CostModel):
    def get_cost_function(self, nodename, op, cost, nodes, child_costs):
        computed = None
        match op:
            case "Nb_Unboxed_Add_Int64":
                computed = 0.1

            case "Nb_Unbox_Int64":
                computed = 0.1

            case "Nb_Box_Int64":
                computed = 0.1

        if computed is not None:
            return computed + sum(child_costs)

        return super().get_cost_function(
            nodename, op, cost, nodes, child_costs
        )


typeinfer_ruleset = (
    basic_ruleset | ruleset_type_infer_add | facts_argument_types
)

if __name__ == "__main__":
    jt = compiler_pipeline(
        add_x_y,
        ruleset=typeinfer_ruleset,
        verbose=True,
        converter_class=ExtendEGraphToRVSDG,
        codegen_extension=codegen_extension,
        cost_model=MyCostModel(),
    )
    res = jt(123, 321)
    run_test(add_x_y, jt, (123, 321), verbose=True)


# ## Optimize boxing logic


def chained_additions(x, y):
    return x + y + y


if __name__ == "__main__":
    jt = compiler_pipeline(
        chained_additions,
        ruleset=typeinfer_ruleset,
        verbose=True,
        converter_class=ExtendEGraphToRVSDG,
        codegen_extension=codegen_extension,
        cost_model=MyCostModel(),
    )
    res = jt(123, 321)
    run_test(chained_additions, jt, (123, 321), verbose=True)

# Improve the rules


@ruleset
def ruleset_optimize_boxing(io: Term, x: Term, y: Term, add: Term):
    yield rewrite(Nb_Box_Int64(Nb_Unbox_Int64(x)), subsume=True).to(x)
    yield rewrite(Nb_Unbox_Int64(Nb_Box_Int64(x)), subsume=True).to(x)


optimized_ruleset = (
    typeinfer_ruleset | ruleset_optimize_boxing
)  # <---- new rule

if __name__ == "__main__":

    jt = compiler_pipeline(
        chained_additions,
        ruleset=optimized_ruleset,
        verbose=True,
        converter_class=ExtendEGraphToRVSDG,
        codegen_extension=codegen_extension,
        cost_model=MyCostModel(),
    )
    res = jt(123, 321)
    run_test(chained_additions, jt, (123, 321), verbose=True)
