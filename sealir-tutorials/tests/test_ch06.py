from sealir import ase

from ch06_mlir_backend import *


def test_ch06_example_1():
    llvm_module, func_egraph = compiler.lower_py_fn(
        example_1,
        argtypes=(Int64, Int64),
        ruleset=(if_else_ruleset | setup_argtypes(TypeInt64, TypeInt64)),
    )
    compiler.run_backend_passes(llvm_module)
    jt = compiler.compile_module(llvm_module, func_egraph)

    args = (10, 33)
    run_test(example_1, jt, args, verbose=False)
    args = (7, 3)
    run_test(example_1, jt, args, verbose=False)


def test_ch06_example_2():
    llvm_module, func_egraph = compiler.lower_py_fn(
        example_2,
        argtypes=(Int64, Int64),
        ruleset=(
            if_else_ruleset
            | setup_argtypes(TypeInt64, TypeInt64)
            | ruleset_type_infer_float  # < --- added for float()
        ),
    )
    compiler.run_backend_passes(llvm_module)
    jt = compiler.compile_module(llvm_module, func_egraph)

    args = (10, 33)
    run_test(example_2, jt, args, verbose=False)
    args = (7, 3)
    run_test(example_2, jt, args, verbose=False)


def test_ch06_example_3():
    llvm_module, func_egraph = compiler.lower_py_fn(
        example_3,
        argtypes=(Int64, Int64),
        ruleset=loop_ruleset | setup_argtypes(TypeInt64, TypeInt64),
    )
    compiler.run_backend_passes(llvm_module)
    jt = compiler.compile_module(llvm_module, func_egraph)

    run_test(example_3, jt, (10, 7), verbose=False)


def test_ch06_example_4():
    llvm_module, func_egraph = compiler.lower_py_fn(
        example_4,
        argtypes=(Int64, Int64),
        ruleset=loop_ruleset | setup_argtypes(TypeInt64, TypeInt64),
    )
    compiler.run_backend_passes(llvm_module)
    jt = compiler.compile_module(llvm_module, func_egraph)

    run_test(example_4, jt, (10, 7), verbose=False)
