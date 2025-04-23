# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Ch 5. Type infer array operations
#

from __future__ import annotations

import ctypes

import numpy as np
import sealir.rvsdg.grammar as rg
from egglog import (
    EGraph,
    Expr,
    Ruleset,
    String,
    StringLike,
    Unit,
    Vec,
    birewrite,
    eq,
    function,
    i64,
    i64Like,
    join,
    method,
    ne,
    rewrite,
    rule,
    ruleset,
    set_,
    union,
    vars_,
)
from llvmlite import ir
from sealir.eqsat.py_eqsat import (
    Py_SubscriptIO,
)
from sealir.eqsat.rvsdg_eqsat import (
    GraphRoot,
    InPorts,
    Port,
    PortList,
    Region,
    Term,
    TermList,
    wildcard,
)

from ch04_1_typeinfer_ifelse import (
    Grammar,
    NbOp_Type,
    TypedIns,
    _wc,
)
from ch04_2_typeinfer_loops import Backend as _ch04_2_Backend
from ch04_2_typeinfer_loops import (
    ExtendEGraphToRVSDG as _ch04_2_ExtendEGraphToRVSDG,
)
from ch04_2_typeinfer_loops import (
    Int64,
    MyCostModel,
    NbOp_Base,
    SExpr,
    Type,
    TypeInt64,
    TypeVar,
    base_ruleset,
    compiler_pipeline,
    run_test,
)
from utils import IN_NOTEBOOK

# ## Define the `ArrayDesc` to describe metadata for an Array type


# Define DataLayout


class Dim(Expr):
    @classmethod
    def fixed(self, size: i64) -> Dim: ...
    @classmethod
    def symbolic(self, unque_id: StringLike) -> Dim: ...


# Define Dim for the shape info at each dimension


class DataLayout(Expr):
    @classmethod
    def c_contiguous(cls) -> DataLayout: ...
    @classmethod
    def fortran_contiguous(cls) -> DataLayout: ...
    @classmethod
    def strided(cls) -> DataLayout: ...


# Define ArrayDesc


class ArrayDesc(Expr):
    def __init__(self, uid: StringLike): ...

    @property
    def dtype(self) -> Type: ...

    @property
    def ndim(self) -> i64: ...

    def dim(self, idx: i64Like) -> Dim: ...

    @property
    def dataLayout(self) -> DataLayout: ...

    def toType(self) -> Type: ...


if __name__ == "__main__":

    array0 = ArrayDesc(uid="array0")
    eg = EGraph()
    eg.let("array0", array0)
    eg.register(set_(array0.dtype).to(TypeInt64))
    if IN_NOTEBOOK:
        eg.display(graphviz=True)

if __name__ == "__main__":
    eg.register(
        set_(array0.ndim).to(3),
        set_(array0.dim(0)).to(Dim.symbolic("M")),
        set_(array0.dim(1)).to(Dim.symbolic("N")),
        set_(array0.dim(2)).to(Dim.fixed(4)),
    )
    if IN_NOTEBOOK:
        eg.display(graphviz=True)

if __name__ == "__main__":
    eg.register(
        set_(array0.dataLayout).to(DataLayout.c_contiguous()),
    )
    if IN_NOTEBOOK:
        eg.display(graphviz=True)


# ### Merging symbolic dimension to fixed dimension

# introduce a new array `array1`

if __name__ == "__main__":
    array1 = ArrayDesc(uid="array1")
    eg.register(
        set_(array1.ndim).to(3),
        set_(array1.dim(0)).to(Dim.fixed(10)),
        set_(array1.dim(1)).to(Dim.fixed(24)),
        set_(array1.dim(2)).to(Dim.symbolic("K")),
    )
    if IN_NOTEBOOK:
        eg.display(graphviz=True)


# merge `array0` with `array1`

if __name__ == "__main__":
    eg.register(union(array0).with_(array1))
    eg.run(1)
    if IN_NOTEBOOK:
        eg.display(graphviz=True)

    # check that the Dim merged
    eg.check(array0.dim(0) == array1.dim(0))
    eg.check(array0.dim(1) == array1.dim(1))
    eg.check(array0.dim(2) == array1.dim(2))

    eg.check(Dim.symbolic("M") == Dim.fixed(10))
    eg.check(Dim.symbolic("N") == Dim.fixed(24))
    eg.check(Dim.symbolic("K") == Dim.fixed(4))

# ## Extend the compiler


class NbOp_ArrayDimFixed(NbOp_Base):
    size: int


class NbOp_ArrayDimSymbolic(NbOp_Base):
    name: str


class NbOp_ArrayType(NbOp_Base):
    dtype: NbOp_Type
    ndim: int
    datalayout: str
    shape: tuple[SExpr, ...]


# ## Example 1: ``Array.__getitem__``


def example_1(ary, idx):
    return ary[idx]


array_1d_symbolic = NbOp_ArrayType(
    dtype=Int64,
    ndim=1,
    datalayout="c_contiguous",
    shape=(NbOp_ArrayDimSymbolic("m"),),
)


@ruleset
def facts_function_types(
    outports: Vec[Port],
    func_uid: String,
    reg_uid: String,
    fname: String,
    region: Region,
):
    array_int64_1d = ArrayDesc(uid="array_int64_1d")
    yield rule(array_int64_1d).then(
        set_(array_int64_1d.ndim).to(i64(1)),
        set_(array_int64_1d.dim(0)).to(Dim.symbolic("n")),
        set_(array_int64_1d.dtype).to(TypeInt64),
        set_(array_int64_1d.dataLayout).to(DataLayout.c_contiguous()),
    )

    yield rule(
        # This match the function at graph root
        GraphRoot(
            Term.Func(
                body=Term.RegionEnd(region=region, ports=PortList(outports)),
                uid=func_uid,
                fname=fname,
            )
        ),
        region == Region(uid=reg_uid, inports=_wc(InPorts)),
    ).then(
        # The first argument is Int64
        set_(TypedIns(region).arg(1).getType()).to(array_int64_1d.toType()),
        # The second argument is Int64
        set_(TypedIns(region).arg(2).getType()).to(TypeInt64),
    )


@ruleset
def ruleset_typeinfer_array_getitem(
    getitem: Term,
    io: Term,
    ary: Term,
    index: Term,
    ty: Type,
    ary_uid: String,
    arydesc: ArrayDesc,
    itemty: Type,
):
    yield rule(
        # Implement getitem(int)->scalar
        getitem == Py_SubscriptIO(io, ary, index),
        # ary is array type
        ty == TypeVar(ary).getType(),
        ty == arydesc.toType(),
        # index is int type
        TypeVar(index).getType() == TypeInt64,
        # then ary must be 1D
        arydesc.ndim == i64(1),
        # get item type
        itemty == arydesc.dtype,
    ).then(
        # shortcut IO
        union(getitem.getPort(0)).with_(io),
        # Rewrite operation
        union(getitem.getPort(1)).with_(
            Nb_Array_1D_Getitem_Scalar(io, ary, index, itemty)
        ),
        # Return type is int64
        set_(TypeVar(getitem.getPort(1)).getType()).to(itemty),
    )


@function
def Nb_Array_1D_Getitem_Scalar(
    io: Term, ary: Term, index: Term, dtype: Type
) -> Term: ...


class NbOp_Array_1D_Getitem_Scalar(NbOp_Base):
    io: SExpr
    ary: SExpr
    index: SExpr
    attr: SExpr


class ExtendEGraphToRVSDG(_ch04_2_ExtendEGraphToRVSDG):
    def handle_Term(self, op: str, children: dict | list, grm: Grammar):
        match op, children:
            case "Nb_Array_1D_Getitem_Scalar", {
                "io": io,
                "ary": ary,
                "index": index,
                "dtype": dtype,
            }:
                return grm.write(
                    NbOp_Array_1D_Getitem_Scalar(
                        io=io,
                        ary=ary,
                        index=index,
                        attr=grm.write(rg.Attrs(dtype)),
                    )
                )
        return super().handle_Term(op, children, grm)


class Backend(_ch04_2_Backend):

    def lower_type(self, ty: NbOp_Type):
        match ty:
            case NbOp_ArrayType(
                dtype=dtype,
                ndim=int(ndim),
                datalayout=str(datalayout),
                shape=shape,
            ):
                ll_dtype = self.lower_type(dtype)
                ptr = ll_dtype.as_pointer()
                shape_array = ir.ArrayType(ir.IntType(64), ndim)
                return ir.LiteralStructType([ptr, shape_array]).as_pointer()

        return super().lower_type(ty)

    def lower_expr(self, expr, state):
        builder = state.builder
        match expr:
            case NbOp_Array_1D_Getitem_Scalar(
                io=io, ary=ary, index=index, attr=attr
            ):
                io = yield io
                ary = yield ary
                index = yield index
                match attr:
                    case rg.Attrs((NbOp_Type(str(typename)),)):
                        pass
                    case _:
                        assert False, attr
                arystruct = builder.load(ary)
                dataptr = builder.extract_value(arystruct, 0)
                ptr_offset = builder.gep(dataptr, [index])
                return builder.load(ptr_offset)

        return (yield from super().lower_expr(expr, state))

    def get_ctype(self, lltype: ir.Type):
        match lltype:
            case ir.PointerType():
                return ctypes.c_void_p()

        return super().get_ctype(lltype)


class CtypeInt64Array1D(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p), ("shape", (ctypes.c_uint64 * 1))]


if __name__ == "__main__":
    jt = compiler_pipeline(
        example_1,
        argtypes=(array_1d_symbolic, Int64),
        ruleset=(
            base_ruleset
            | facts_function_types
            | ruleset_typeinfer_array_getitem
        ),
        verbose=True,
        converter_class=ExtendEGraphToRVSDG,
        cost_model=MyCostModel(),
        backend=Backend(),
    )
    ary = np.arange(10, dtype=np.int64)
    param_ary = CtypeInt64Array1D()
    param_ary.ptr = ary.ctypes.data
    param_ary.shape[0] = ary.shape[0]

    got = jt(ctypes.byref(param_ary), 3)
    print("got", got)
    expect = example_1(ary, 3)
    assert got == expect


# ## Example 2: Sum numbers in 1D array


def example_2(ary, size):
    i = 0
    c = 0
    while i < size:
        c = c + ary[i]
        i = i + 1
    return c


if __name__ == "__main__":
    jt = compiler_pipeline(
        example_2,
        argtypes=(array_1d_symbolic, Int64),
        ruleset=(
            base_ruleset
            | facts_function_types
            | ruleset_typeinfer_array_getitem
        ),
        verbose=True,
        converter_class=ExtendEGraphToRVSDG,
        cost_model=MyCostModel(),
        backend=Backend(),
    )
    ary = np.arange(10, dtype=np.int64)
    param_ary = CtypeInt64Array1D()
    param_ary.ptr = ary.ctypes.data
    param_ary.shape[0] = ary.shape[0]

    got = jt(ctypes.byref(param_ary), ary.size)
    print("got", got)
    expect = example_2(ary, ary.size)
    assert got == expect
