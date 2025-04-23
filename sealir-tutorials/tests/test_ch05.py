from sealir import ase

from ch05_typeinfer_array import *


def test_ch5_example_1():
    jt = compiler_pipeline(
        example_1,
        argtypes=(array_1d_symbolic, Int64),
        ruleset=(
            base_ruleset
            | facts_function_types
            | ruleset_typeinfer_array_getitem
        ),
        verbose=False,
        converter_class=ExtendEGraphToRVSDG,
        cost_model=MyCostModel(),
        backend=Backend(),
    )
    ary = np.arange(10, dtype=np.int64)
    param_ary = CtypeInt64Array1D()
    param_ary.ptr = ary.ctypes.data
    param_ary.shape[0] = ary.shape[0]

    got = jt(ctypes.byref(param_ary), 3)
    expect = example_1(ary, 3)
    assert got == expect


def test_ch5_example_2():
    jt = compiler_pipeline(
        example_2,
        argtypes=(array_1d_symbolic, Int64),
        ruleset=(
            base_ruleset
            | facts_function_types
            | ruleset_typeinfer_array_getitem
        ),
        verbose=False,
        converter_class=ExtendEGraphToRVSDG,
        cost_model=MyCostModel(),
        backend=Backend(),
    )
    ary = np.arange(10, dtype=np.int64)
    param_ary = CtypeInt64Array1D()
    param_ary.ptr = ary.ctypes.data
    param_ary.shape[0] = ary.shape[0]

    got = jt(ctypes.byref(param_ary), ary.size)
    expect = example_2(ary, ary.size)
    assert got == expect
