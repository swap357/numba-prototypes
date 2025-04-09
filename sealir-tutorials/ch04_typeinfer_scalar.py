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
#
# In this chapter, we'll add type inference logic into the EGraph middle-end.

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

# First, we need some modifications to the compiler-pipeline.
# The middle-end is augmented with the following:
#
# - `converter_class` is for customizing EGraph-to-RVSDG conversion as we will be
#   introducing new RVSDG operations for typed operations.
# - `cost_model` is for customizing the cost of the new operations.


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
        converter_class=converter_class,  # <---- new
        cost_model=cost_model,  # <---- new
    )
    return cost, extracted


# The compiler_pipeline will have a `codegen_extension` for defining LLVM
# code-generation for the new operations.


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


# ## A Simple Type Inference Example

# First, we will start with a simple binary add operation.


def add_x_y(x, y):
    return x + y


# We will start with the same ruleset as in chapter 3.

basic_ruleset = rvsdg_eqsat.ruleset_rvsdg_basic | ruleset_const_propagate


# We will test our base compiler (ch 3 compiler behavior) on our function to set
# the baseline. At this stage, no type inference is happening.

if __name__ == "__main__":
    # start with previous compiler pipeline
    jt = compiler_pipeline(add_x_y, ruleset=basic_ruleset, verbose=True)
    run_test(add_x_y, jt, (123, 321), verbose=True)


# ### Adding type inference
#
# A new EGraph expression class (`Expr`) is added to represent type:


class Type(Expr):
    def __init__(self, name: StringLike): ...


# Then, we add a EGraph function to determine the type-of a `Term`:


@function
def TypeOf(x: Term) -> Type: ...


# Next, we define functions for the new operations:
#
# - `Nb_Unbox_Int64` unboxes a PyObject into a Int64.
# - `Nb_Box_Int64` boxes a Int64 into a PyObject.
# - `Nb_Unboxed_Add_Int64` performs a Int64 addition on unboxed operands.


@function
def Nb_Unbox_Int64(val: Term) -> Term: ...
@function
def Nb_Box_Int64(val: Term) -> Term: ...
@function
def Nb_Unboxed_Add_Int64(lhs: Term, rhs: Term) -> Term: ...


# Now, we define the first type-inference rule:
#
# If a `Py_AddIO()` (a Python binary add operation) is applied to operands
# that are known Int64, convert it into the unboxed add. The output type will
# be Int64. The IO state into the `Py_AddIO()` will be unchanged.

# +
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


# -

# The following rule defines some fact about the function being compiled.
# It declares that the two arguments are Int64.


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
        GraphRoot(
            Term.Func(
                body=Term.RegionEnd(region=region, ports=outports),
                uid=func_uid,
                fname=fname,
            )
        ),
        arg_x == region.get(1),
        arg_y == region.get(2),
    ).then(
        union(TypeOf(arg_x)).with_(TypeInt64),
        union(TypeOf(arg_y)).with_(TypeInt64),
    )


# ### Defining conversion into RVSDG

# We will expand the RVSDG grammar with the typed operations.
#
# Each of the new typed operations will require a corresponding grammar rule.

# +
SExpr = rvsdg.grammar.SExpr


class NbOp_Base(grammar.Rule):
    pass


class NbOp_Unboxed_Add_Int64(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Unbox_Int64(NbOp_Base):
    val: SExpr


class NbOp_Box_Int64(NbOp_Base):
    val: SExpr


# -

# The new grammar for our IR is a combination of the new typed-operation grammar
# and the base RVSDG grammar.


class Grammar(grammar.Grammar):
    start = rvsdg.Grammar.start | NbOp_Base


# Now, we define a EGraph-to-RVSDG conversion class that is expanded to handle
# the new grammar.


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
                # Use parent's implementation for other terms.
                return super().handle_Term(op, children, grm)


# The LLVM code-generation also needs an extension:


def codegen_extension(expr, args, builder, pyapi):
    match expr._head, args:
        case "NbOp_Unboxed_Add_Int64", (lhs, rhs):
            return SSAValue(builder.add(lhs.value, rhs.value))
        case "NbOp_Unbox_Int64", (val,):
            return SSAValue(pyapi.long_as_longlong(val.value))
        case "NbOp_Box_Int64", (val,):
            return SSAValue(pyapi.long_from_longlong(val.value))
    return NotImplemented


# A new cost model to prioritize the typed operations:


class MyCostModel(CostModel):
    def get_cost_function(self, nodename, op, cost, nodes, child_costs):
        self_cost = None
        match op:
            case "Nb_Unboxed_Add_Int64":
                self_cost = 0.1

            case "Nb_Unbox_Int64":
                self_cost = 0.1

            case "Nb_Box_Int64":
                self_cost = 0.1

        if self_cost is not None:
            return self_cost + sum(child_costs)

        # Fallthrough to parent's cost function
        return super().get_cost_function(
            nodename, op, cost, nodes, child_costs
        )


# The new ruleset with the type inference logic and facts about the compiled
# function:

typeinfer_ruleset = (
    basic_ruleset | ruleset_type_infer_add | facts_argument_types
)

# We are now ready to run the compiler:

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


# Observations:
#
# - In the egraph, observe how the new operations are represented.
# - In the RVSDG, notice the lack of `Py_AddIO()`
# - In the LLVM, notice the addition is now done in native `i64`.

# ## Optimize boxing logic


# A key benefit of EGraph is that there is no need to specify ordering to
# "compiler-passes". To demonstrate this, we will insert optimization rules
# on the boxing and unboxing operation. `unbox(box(x))` is equivalent
# to an no-op. We can remove redundant boxing and unboxing.

# We will need more than one addition to showcase the optimization:


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

# Observations:
#
# ```
#   $4 = NbOp_Box_Int64 $3
#   $5 = NbOp_Unbox_Int64 $4
# ```
#
# The box and unbox chain is redundant (i.e. `$3 = $5`).

# ### Box/Unbox optimization rules


# The needed optimization rule is very simple. Any chained box-unbox; or unbox-box
# are redundant.
#
# (We use `subsume=True` to delete the original EGraph node (enode) to shrink
# the graph early.)


@ruleset
def ruleset_optimize_boxing(x: Term):
    yield rewrite(Nb_Box_Int64(Nb_Unbox_Int64(x)), subsume=True).to(x)
    yield rewrite(Nb_Unbox_Int64(Nb_Box_Int64(x)), subsume=True).to(x)


optimized_ruleset = typeinfer_ruleset | ruleset_optimize_boxing

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

# Observations:
#
# ```
#   $1 = NbOp_Unbox_Int64 $0[1]
#   $2 = NbOp_Unbox_Int64 $0[2]
#   $3 = NbOp_Unboxed_Add_Int64 $1 $2
#   $4 = NbOp_Unboxed_Add_Int64 $3 $2
#   $5 = NbOp_Box_Int64 $4
# ```
#
# There is no redundant box-unbox between the two unboxed add anymore.

#
