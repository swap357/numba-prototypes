from ch01_basic_compiler import frontend, backend, jit_compile

from sealir import ase, rvsdg

from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_extract import egraph_extraction
from sealir.eqsat.rvsdg_eqsat import GraphRoot

from egglog import EGraph


def middle_end(rvsdg_expr, apply_to_egraph):
    """The middle end encode the RVSDG into a EGraph to apply rewrite rules.
    After that, it is extracted back into RVSDG.
    """
    # Convert to egraph
    memo = egraph_conversion(rvsdg_expr)
    root = GraphRoot(memo[rvsdg_expr])

    egraph = EGraph()
    egraph.let("root", root)

    apply_to_egraph(egraph, root)

    # Extraction
    cost, extracted = egraph_extraction(egraph, rvsdg_expr)
    return cost, extracted


def main():
    def sum_ints(n):
        c = 1 + n
        # WIP: fixing up SealIR to do loops
        #     for i in range(n):
        #         c += i
        #     return c
        return c

    rvsdg_expr, dbginfo = frontend(sum_ints)


    # Middle end
    def display_egraph(egraph, root):
        egraph.display()

    cost, extracted = middle_end(rvsdg_expr, display_egraph)
    print("Extracted from EGraph".center(80, '='))
    print("cost =", cost)
    print(rvsdg.format_rvsdg(extracted))

    llmod = backend(rvsdg_expr)

    jt = jit_compile(llmod, rvsdg_expr)
    res = jt(12)

    print("JIT: output".center(80, '='))
    print(res)


if __name__ == "__main__":
    main()