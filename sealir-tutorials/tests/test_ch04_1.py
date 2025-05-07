from sealir import ase

from ch04_1_typeinfer_ifelse import *


def test_ch04_1_example_1():
    jt = compiler_pipeline(
        example_1,
        argtypes=(Int64, Int64),
        ruleset=(base_ruleset | setup_argtypes(TypeInt64, TypeInt64)),
        verbose=False,
        converter_class=ExtendEGraphToRVSDG,
        cost_model=MyCostModel(),
        backend=Backend(),
    )
    args = (10, 33)
    run_test(example_1, jt, args, verbose=False)
    args = (7, 3)
    run_test(example_1, jt, args, verbose=False)


def test_ch04_1_example_2():
    jt = compiler_pipeline(
        example_2,
        argtypes=(Int64, Int64),
        ruleset=(
            base_ruleset
            | setup_argtypes(TypeInt64, TypeInt64)
            | ruleset_type_infer_float  # < --- added for float()
        ),
        verbose=False,
        converter_class=ExtendEGraphToRVSDG,
        cost_model=MyCostModel(),
        backend=Backend(),
    )
    args = (10, 33)
    run_test(example_2, jt, args, verbose=False)
    args = (7, 3)
    run_test(example_2, jt, args, verbose=False)


def test_ch04_1_example_3():
    try:
        compiler_pipeline(
            example_3,
            argtypes=(Int64, Int64),
            ruleset=(
                base_ruleset
                | setup_argtypes(TypeInt64, TypeInt64)
                | ruleset_type_infer_float
                | ruleset_failed_to_unify
            ),
            verbose=False,
            converter_class=ExtendEGraphToRVSDG,
            cost_model=MyCostModel(),
            backend=Backend(),
        )
    except CompilationError as e:
        # Compilation failed because the return type cannot be determined.
        # This indicates that the type inference is incomplete
        print_exception(e)
        assert "fail to unify" in str(e)


def test_ch04_1_example_4():
    try:
        compiler_pipeline(
            example_3,
            argtypes=(Int64, Int64),
            ruleset=(
                base_ruleset
                | setup_argtypes(TypeInt64, TypeInt64)
                | ruleset_type_infer_float
                | ruleset_failed_to_unify
                | ruleset_type_infer_failure_report
            ),
            verbose=False,
            converter_class=ExtendEGraphToRVSDG,
            cost_model=MyCostModel(),
            backend=Backend(),
        )

    except CompilationError as e:
        print_exception(e)
        assert "Failed to unify if-else outgoing variables: z" in str(e)
