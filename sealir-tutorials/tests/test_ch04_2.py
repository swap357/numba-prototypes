from sealir import ase

from ch04_2_typeinfer_loops import *


def test_ch04_2_example_1():
    llvm_module, func_egraph = compiler.lower_py_fn(
        example_1,
        argtypes=(Int64, Int64),
        ruleset=base_ruleset | setup_argtypes(TypeInt64, TypeInt64),
    )
    jt = compiler.compile_module(llvm_module, func_egraph)
    run_test(example_1, jt, (10, 7), verbose=False)


def test_ch04_2_example_2():
    llvm_module, func_egraph = compiler.lower_py_fn(
        example_2,
        argtypes=(Int64, Int64),
        ruleset=base_ruleset | setup_argtypes(TypeInt64, TypeInt64),
    )
    jt = compiler.compile_module(llvm_module, func_egraph)
    run_test(example_2, jt, (10, 7), verbose=False)
