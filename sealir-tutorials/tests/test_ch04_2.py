from sealir import ase

from ch04_2_typeinfer_loops import *


def test_ch04_2_example_1():
    jt = compiler_pipeline(
        example_1,
        argtypes=(Int64, Int64),
        ruleset=base_ruleset | setup_argtypes(TypeInt64, TypeInt64),
        verbose=False,
        converter_class=ExtendEGraphToRVSDG,
        cost_model=MyCostModel(),
        backend=Backend(),
    )
    run_test(example_1, jt, (10, 7), verbose=False)


def test_ch04_2_example_2():
    jt = compiler_pipeline(
        example_2,
        argtypes=(Int64, Int64),
        ruleset=base_ruleset | setup_argtypes(TypeInt64, TypeInt64),
        verbose=False,
        converter_class=ExtendEGraphToRVSDG,
        cost_model=MyCostModel(),
        backend=Backend(),
    )
    run_test(example_2, jt, (10, 7), verbose=False)
