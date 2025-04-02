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

# # Ch 3. EGraph Program Rewrites
#
# In this chapter, we will implement our first program rewrite.

from __future__ import annotations

from egglog import EGraph, Unit, function, i64, rewrite, rule, ruleset
from sealir import rvsdg
from sealir.eqsat import rvsdg_eqsat
from sealir.eqsat.rvsdg_eqsat import GraphRoot, Term, TermList

# We'll be extending from chapter 2.
from ch02_egraph_basic import (
    backend,
    frontend,
    jit_compile,
    middle_end,
    run_test,
)

# Here's a new compiler pipeline with customizable rulesets.
# A new `ruleset` argument is added.


def compiler_pipeline(fn, *, verbose=False, ruleset):
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

    cost, extracted = middle_end(rvsdg_expr, define_egraph)
    print("Extracted from EGraph".center(80, "="))
    print("cost =", cost)
    print(rvsdg.format_rvsdg(extracted))

    llmod = backend(rvsdg_expr)
    return jit_compile(llmod, rvsdg_expr)


# ## Rules for defining constants
#
# Starting with a simple rule, we will define what is a constant boolean.
# To do so, they are setup as an `egglog.function`.
# It will acts as a fact on a `Term` if the `Term` is an expression of a literal
# int64.


# +
@function
def IsConstantTrue(t: Term) -> Unit: ...


@function
def IsConstantFalse(t: Term) -> Unit: ...


# -


@ruleset
def ruleset_const_propagate(a: Term, ival: i64):
    # a Literal Int64 is constant True if it's non-zero
    yield rule(
        # Given a LiteralI64 where the integer-value is non zero
        a == Term.LiteralI64(ival),
        ival != 0,
    ).then(
        # Setup the following fact
        IsConstantTrue(a)
    )
    # a Literal Int64 is constant False if it's zero
    yield rule(
        # Given a LiteralI64 where the integer-value is zero
        a == Term.LiteralI64(ival),
        ival == 0,
    ).then(
        # Setup the following fact
        IsConstantFalse(a)
    )


if __name__ == "__main__":

    def ifelse_fold(a, b):
        c = 0
        if c:
            return a
        else:
            return b

    # Add our const-propagation rule to the basic rvsdg ruleset
    my_ruleset = rvsdg_eqsat.ruleset_rvsdg_basic | ruleset_const_propagate

    jt = compiler_pipeline(ifelse_fold, verbose=True, ruleset=my_ruleset)
    run_test(ifelse_fold, jt, (12, 34))


# In the above, notice there's a new node for `IsConstantFalse` on the
# `LiteralI64(0)`. That shows it is successfully finding constants.

# ## Rules for folding if-else
#
# Let's make a more involved rule. This time we will fold if-else expression
# that has constant condition.


@ruleset
def ruleset_const_fold_if_else(a: Term, b: Term, c: Term, operands: TermList):
    yield rewrite(
        # Define the if-else pattern to match
        Term.IfElse(cond=a, then=b, orelse=c, operands=operands),
        subsume=True,  # subsume to disable extracting the original term
    ).to(
        # Define the target expression
        # This apply region `b` (then) using the `operands`.
        Term.Apply(b, operands),
        # Given that the condition is constant True
        IsConstantTrue(a),
    )
    yield rewrite(
        # Define the if-else pattern to match
        Term.IfElse(cond=a, then=b, orelse=c, operands=operands),
        subsume=True,  # subsume to disable extracting the original term
    ).to(
        # Define the target expression.
        # This apply region `c` (orelse) using the `operands`.
        Term.Apply(c, operands),
        # Given that the condition is constant False
        IsConstantFalse(a),
    )


if __name__ == "__main__":
    my_ruleset = (
        rvsdg_eqsat.ruleset_rvsdg_basic
        | ruleset_const_propagate
        | ruleset_const_fold_if_else  # <-- the new rule for if-else
    )

    jt = compiler_pipeline(ifelse_fold, verbose=True, ruleset=my_ruleset)
    run_test(ifelse_fold, jt, (12, 34))

# After the rewrite, the RVSDG is simplied to a mostly empty function body. The
# `!ret` is hard coded to `$0[2]` which is the variable `b`, corresponding to
# the `return b` in the `else` branch.
#
# The egraph has become more interesting with many nodes merged, indicating
# they are equivalent. For instance, the `Term.Apply` and `Term.IfElse` nodes
# are merged.
#
