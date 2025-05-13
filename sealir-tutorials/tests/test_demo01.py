from demo01_gelu_tanh_approx import *


def test_demo01_baseline():
    be = Backend()
    jt = compiler_pipeline(
        gelu_tanh_forward,
        argtypes=(Float32,),
        ruleset=(
            base_ruleset | setup_argtypes(TypeFloat32) | additional_rules
        ),
        converter_class=ExtendEGraphToRVSDG,
        cost_model=MyCostModel(),
        backend=be,
    )
    assert "llvm.call @tanhf" in str(be.module)
    run_test(gelu_tanh_forward, jt, (0.234,))


def test_demo01_optimized():
    be = Backend()

    jt = compiler_pipeline(
        gelu_tanh_forward,
        argtypes=(Float32,),
        ruleset=(
            base_ruleset
            | setup_argtypes(TypeFloat32)
            | additional_rules
            | optimize_rules
        ),
        converter_class=ExtendEGraphToRVSDG,
        cost_model=MyCostModel(),
        backend=be,
    )
    # tanhf not used
    assert "llvm.call @tanhf" not in str(be.module)
    # powf not used
    assert "llvm.call @powf" not in str(be.module)
    # test correctness
    relclose = lambda x, y: np.allclose(x, y, rtol=1e-6)
    run_test(gelu_tanh_forward, jt, (0.234,), equal=relclose)
