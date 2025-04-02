from ch02_egraph_basic import *


def test_ch02_sum_ints():
    def sum_ints(n):
        c = 0
        for i in range(n):
            c += i
        return c

    jt = compiler_pipeline(sum_ints)
    run_test(sum_ints, jt, (12,))


def test_ch02_max_two():
    def max_if_else(x, y):
        if x > y:
            return x
        else:
            return y

    jt = compiler_pipeline(max_if_else)
    args = (1, 2)
    run_test(max_if_else, jt, args)
    args = (3, 2)
    run_test(max_if_else, jt, args)
