from sealir import ase

from ch05_typeinfer_array import *


def test_ch5_example_1():
    llvm_module, func_egraph = compiler.lower_py_fn(
        example_1,
        argtypes=(array_1d_symbolic, Int64),
        ruleset=(
            base_ruleset
            | setup_argtypes(array_int64_1d.toType(), TypeInt64)
            | ruleset(*array_infos)
            | ruleset_typeinfer_array_getitem
        ),
    )
    jt = compiler.compile_module(llvm_module, func_egraph)
    ary = np.arange(10, dtype=np.int64)
    param_ary = CtypeInt64Array1D()
    param_ary.ptr = ary.ctypes.data
    param_ary.shape[0] = ary.shape[0]

    got = jt(ctypes.byref(param_ary), 3)
    expect = example_1(ary, 3)
    assert got == expect


def test_ch5_example_2():
    llvm_module, func_egraph = compiler.lower_py_fn(
        example_2,
        argtypes=(array_1d_symbolic, Int64),
        ruleset=(
            base_ruleset
            | setup_argtypes(array_int64_1d.toType(), TypeInt64)
            | ruleset(*array_infos)
            | ruleset_typeinfer_array_getitem
        ),
    )
    jt = compiler.compile_module(llvm_module, func_egraph)

    ary = np.arange(10, dtype=np.int64)
    param_ary = CtypeInt64Array1D()
    param_ary.ptr = ary.ctypes.data
    param_ary.shape[0] = ary.shape[0]

    got = jt(ctypes.byref(param_ary), ary.size)
    expect = example_2(ary, ary.size)
    assert got == expect


def test_broadcasting():
    eg = EGraph()

    # array0 is M x N x 10 x 4
    array0 = ArrayDesc(uid="array0")
    eg.register(
        set_(array0.dtype).to(TypeInt64),
        set_(array0.ndim).to(4),
        set_(array0.dim(0)).to(Dim.symbolic("M")),
        set_(array0.dim(1)).to(Dim.symbolic("N")),
        set_(array0.dim(2)).to(Dim.fixed(10)),
        set_(array0.dim(3)).to(Dim.fixed(4)),
    )

    # array1 is 1 x 4
    array1 = ArrayDesc(uid="array1")
    eg.let("array1", array1)
    eg.register(
        set_(array1.dtype).to(TypeInt64),
        set_(array1.ndim).to(2),
        set_(array1.dim(0)).to(Dim.fixed(1)),
        set_(array1.dim(1)).to(Dim.fixed(4)),
    )

    broadcasted = Broadcast(array0, array1)
    eg.let("broadcasted", broadcasted)

    eg.run(ruleset_broadcasting.saturate())
    # Verify
    eg.check(broadcasted.dim(0) == Dim.symbolic("M"))
    eg.check(broadcasted.dim(1) == Dim.symbolic("N"))
    eg.check(broadcasted.dim(2) == Dim.fixed(10))
    eg.check(broadcasted.dim(3) == Dim.fixed(4))


def test_broadcasting_error():
    eg = EGraph()

    # array0 is 10 x 4
    array0 = ArrayDesc(uid="array0")
    eg.register(
        set_(array0.dtype).to(TypeInt64),
        set_(array0.ndim).to(2),
        set_(array0.dim(0)).to(Dim.fixed(10)),
        set_(array0.dim(1)).to(Dim.fixed(4)),
    )

    # array1 is 2
    array1 = ArrayDesc(uid="array1")
    eg.let("array1", array1)
    eg.register(
        set_(array1.dtype).to(TypeInt64),
        set_(array1.ndim).to(1),
        set_(array1.dim(0)).to(Dim.fixed(2)),
    )

    broadcasted = Broadcast(array0, array1)
    eg.let("broadcasted", broadcasted)

    eg.run((ruleset_broadcasting | ruleset_broadcasting_error).saturate())

    # Verify
    eg.check(broadcasted.dim(0) == Dim.fixed(10))
    eg.check(broadcasted.dim(1) == DimBroadcastFailed(1))


def test_can_broadcast():
    eg = EGraph()

    # array0 is 10 x 4
    array0 = ArrayDesc(uid="array0")
    eg.register(
        set_(array0.dtype).to(TypeInt64),
        set_(array0.ndim).to(2),
        set_(array0.dim(0)).to(Dim.fixed(10)),
        set_(array0.dim(1)).to(Dim.fixed(4)),
    )

    # array1 is 2
    array1 = ArrayDesc(uid="array1")
    eg.let("array1", array1)
    eg.register(
        set_(array1.dtype).to(TypeInt64),
        set_(array1.ndim).to(1),
        set_(array1.dim(0)).to(Dim.fixed(2)),
    )

    # Case 1: broadcasting is illegal
    case1 = CanBroadcast(array0, array1)
    eg.let("can_broadcast_1", case1)
    # Case 2: broadcasting is legal
    case2 = CanBroadcast(array0, array0)
    eg.let("can_broadcast_2", case2)
    eg.run(
        (
            ruleset_broadcasting
            | ruleset_broadcasting_error
            | ruleset_can_broadcast
            | ruleset_condition
        ).saturate()
    )
    if IN_NOTEBOOK:
        eg.display(graphviz=True)
    # Verify
    eg.check(case1 == BoolExpr(False))
    eg.check(case2 == BoolExpr(True))
