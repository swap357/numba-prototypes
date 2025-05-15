import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch01_basic_compiler import *

def test_softmax_numpy():
    def softmax_numpy(x: np.ndarray, axis: int) -> np.ndarray:
        x_max = np.max(x, axis)
        e_x = np.exp(x - x_max)
        sum_e_x = np.sum(e_x, axis)
        return e_x / sum_e_x

    rvsdg_expr, dbg = frontend(softmax_numpy)
    print(rvsdg.format_rvsdg(rvsdg_expr))

    # llmod = backend(rvsdg_expr)
    # jt = jit_compile(llmod, rvsdg_expr)
    # args = (np.array([1.0, 2.0, 0.5, 3.0]),)
    # run_test(softmax_numpy, jt, args)

if __name__ == "__main__":
    test_softmax_numpy()