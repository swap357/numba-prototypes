# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---


# # Ch 1. Basic Compiler
#
# ## AST frontend and LLVM backend

from __future__ import annotations

import builtins
from collections import ChainMap

from llvmlite import binding as llvm
from llvmlite import ir

from sealir import rvsdg, ase
from sealir.rvsdg import grammar as rg
from sealir.llvm_pyapi_backend import (
    _codegen_loop,
    CodegenState,
    CodegenCtx,
    PythonAPI,
    JitCallable,
)

# ## The frontend

def frontend(fn):
    """
    Frontend code is all encapsulated in sealir.rvsdg.restructure_source
    """
    rvsdg_expr, dbginfo = rvsdg.restructure_source(fn)

    return rvsdg_expr, dbginfo


def _determine_arity(root: ase.SExpr) -> int:
    match root:
        case rg.Func(args=rg.Args() as args):
            return len(args.arguments)
        case _:
            raise TypeError(root._head)


def backend(root, ns=builtins.__dict__):
    """
    Emit LLVM using Python C-API.

    root: the RVSDG expression for the function
    ns: is the dictionary of global names. A JIT is assumed. Object pointer for
        each key is used.

    Warning:

    - This is for testing only.
    - Does NOT do proper memory management.
    - Does NOT do proper error handling.
    """

    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    mod = ir.Module()

    ll_byte = ir.IntType(8)
    ll_pyobject_ptr = ll_byte.as_pointer()
    # Make LLVM function
    arity = _determine_arity(root)

    assert arity >= 1
    actual_num_args = arity
    fnty = ir.FunctionType(
        ll_pyobject_ptr, [ll_pyobject_ptr] * actual_num_args
    )
    fn = ir.Function(mod, fnty, name="foo")

    # init entry block and builder
    builder = ir.IRBuilder(fn.append_basic_block())
    retval_slot = builder.alloca(ll_pyobject_ptr)
    builder.store(ll_pyobject_ptr(None), retval_slot)  # init retval to NULL

    bb_main = builder.append_basic_block()
    builder.branch(bb_main)
    builder.position_at_end(bb_main)

    ctx = CodegenCtx(
        llvm_module=mod,
        llvm_func=fn,
        builder=builder,
        pyapi=PythonAPI(builder),
        global_ns=ChainMap(ns, __builtins__),
    )

    # Emit the function body
    ase.traverse(root, _codegen_loop, CodegenState(context=ctx))
    return mod


def jit_compile(mod, rvsdg_expr):
    """JIT compile LLVM module into an executable function for this process."""
    llvm_ir = str(mod)

    # Create JIT
    lljit = llvm.create_lljit_compiler()
    rt = (
        llvm.JITLibraryBuilder()
        .add_ir(llvm_ir)
        .export_symbol("foo")
        .add_current_process()
        .link(lljit, "foo")
    )
    ptr = rt["foo"]
    arity = _determine_arity(rvsdg_expr)
    return JitCallable.from_pointer(rt, ptr, arity)


def compiler_pipeline(fn, args, *, verbose=False):
    rvsdg_expr, dbginfo = frontend(fn)

    if verbose:
        print("Frontend: Debug Info on RVSDG".center(80, "="))
        print(dbginfo.show_sources())

        print("Frontend: RVSDG".center(80, "="))
        print(rvsdg.format_rvsdg(rvsdg_expr))

    llmod = backend(rvsdg_expr)

    if verbose:
        print("Backend: LLVM".center(80, "="))
        print(llmod)

    jt = jit_compile(llmod, rvsdg_expr)
    res = jt(*args)

    if verbose:
        print("JIT: output".center(80, "="))
        print(res)

    assert res == fn(*args)


def test_ch01_sum_ints():
    def sum_ints(n):
        c = 0
        for i in range(n):
            c += i
        return c

    compiler_pipeline(sum_ints, (12,), verbose=False)


def test_ch01_max_two():
    def max_if_else(x, y):
        if x > y:
            return x
        else:
            return y

    compiler_pipeline(max_if_else, (1, 2), verbose=False)
    compiler_pipeline(max_if_else, (3, 2), verbose=False)


def main():
    def sum_ints(n):
        c = 0
        for i in range(n):
            c += i
        return c

    compiler_pipeline(sum_ints, (12,), verbose=True)


if __name__ == "__main__":
    main()
