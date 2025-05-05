from sealir import ase

from ch06_mlir_backend import *


def test_ch06_example_1():
    jt = compiler_pipeline(
        example_1,
        argtypes=(Int64, Int64),
        ruleset=(if_else_ruleset | facts_function_types),
        verbose=False,
        converter_class=ConditionalExtendGraphtoRVSDG,
        cost_model=MyCostModel(),
        backend=Backend(),
    )
    args = (10, 33)
    run_test(example_1, jt, args, verbose=False)
    args = (7, 3)
    run_test(example_1, jt, args, verbose=False)


def test_ch06_example_2():
    jt = compiler_pipeline(
        example_2,
        argtypes=(Int64, Int64),
        ruleset=(
            if_else_ruleset
            | facts_function_types
            | ruleset_type_infer_float  # < --- added for float()
        ),
        verbose=False,
        converter_class=ConditionalExtendGraphtoRVSDG,
        cost_model=MyCostModel(),
        backend=Backend(),
    )
    args = (10, 33)
    run_test(example_2, jt, args, verbose=False)
    args = (7, 3)
    run_test(example_2, jt, args, verbose=False)


def test_ch06_example_3():
    try:
        compiler_pipeline(
            example_3,
            argtypes=(Int64, Int64),
            ruleset=(
                if_else_ruleset
                | facts_function_types
                | ruleset_type_infer_float
                | ruleset_failed_to_unify
            ),
            verbose=False,
            converter_class=ConditionalExtendGraphtoRVSDG,
            cost_model=MyCostModel(),
            backend=Backend(),
        )
    except CompilationError as e:
        # Compilation failed because the return type cannot be determined.
        # This indicates that the type inference is incomplete
        print_exception(e)
        assert "fail to unify" in str(e)


def test_ch06_example_4():
    jt = compiler_pipeline(
        example_4,
        argtypes=(Int64, Int64),
        ruleset=loop_ruleset,
        verbose=False,
        converter_class=LoopExtendEGraphToRVSDG,
        cost_model=MyCostModel(),
        backend=Backend(),
    )
    run_test(example_4, jt, (10, 7), verbose=False)


def test_ch06_example_5():
    jt = compiler_pipeline(
        example_5,
        argtypes=(Int64, Int64),
        ruleset=loop_ruleset,
        verbose=False,
        converter_class=LoopExtendEGraphToRVSDG,
        cost_model=MyCostModel(),
        backend=Backend(),
    )
    run_test(example_5, jt, (10, 7), verbose=False)
