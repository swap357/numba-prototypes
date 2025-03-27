from __future__ import annotations

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


def backend(root, ns=__builtins__.__dict__):
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
    bodynode = root.body
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
        retval_slot=retval_slot,
        ports={},
        global_ns=ns,
    )

    # Emit the function body
    memo = ase.traverse(bodynode, _codegen_loop, CodegenState(context=ctx))

    # Handle return value
    retval = memo[bodynode].value
    builder.ret(retval)

    return mod


def jit_compile(mod, arity):
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
    return JitCallable.from_pointer(rt, ptr, arity)


def main():
    def sum_ints(n):
        c = 0
        for i in range(n):
            c += i
        return c

    rvsdg_expr, dbginfo = frontend(sum_ints)

    print(dbginfo.show_sources())

    print(rvsdg.format_rvsdg(rvsdg_expr))

    llmod = backend(rvsdg_expr)
    print(llmod)

    jt = jit_compile(llmod, _determine_arity(rvsdg_expr))
    res = jt(12)
    print(res)


if __name__ == "__main__":
    main()
