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

# # Ch 7. MLIR ufunc operations
#

# In this chapter, we'll look at type inference for array operations.

from __future__ import annotations

import ctypes

import numpy as np
from egglog import (
    ruleset,
)
from ch04_2_typeinfer_loops import (
    Int64,
    MyCostModel,
    TypeInt64,
    base_ruleset,
    compiler_pipeline,
    setup_argtypes,
)
from ch05_typeinfer_array import NbOp_ArrayType, NbOp_ArrayDimSymbolic, array_desc_rules, ruleset_typeinfer_array_getitem, Term, Type, String, ArrayDesc, rule, TypeVar, union, set_, Grammar, NbOp_Base, rg, function
from ch06_mlir_backend import ConditionalExtendGraphtoRVSDG as _ExtendEGraphToRVSDG, Backend as _Backend, NbOp_Type, SExpr, LowerStates
from sealir.eqsat.py_eqsat import (
    Py_NegIO,
)

import mlir.dialects.arith as arith
import mlir.dialects.math as math
import mlir.dialects.memref as memref
import mlir.dialects.linalg as linalg
import mlir.dialects.cf as cf
import mlir.dialects.func as func
import mlir.dialects.scf as scf
import mlir.execution_engine as execution_engine
import mlir.ir as ir
import mlir.passmanager as passmanager

Float64 = NbOp_Type("Float64")
TypeFloat64 = Type.simple("Float64")

array_2d_symbolic = NbOp_ArrayType(
    dtype=Float64,
    ndim=2,
    datalayout="c_contiguous",
    shape=(NbOp_ArrayDimSymbolic("m"), NbOp_ArrayDimSymbolic("n")),
)


@ruleset
def ruleset_typeinfer_array_neg(
    src_ary: Term,
    io: Term,
    ary: Term,
    ty: Type,
    ary_uid: String,
    arydesc: ArrayDesc,
    itemty: Type,
):
    yield rule(
        # Implement getitem(int)->scalar
        src_ary == Py_NegIO(io, ary),
        # ary is array type
        ty == TypeVar(ary).getType(),
        ty == arydesc.toType(),
        # get item type
        itemty == arydesc.dtype,
    ).then(
        # shortcut IO
        union(src_ary.getPort(0)).with_(io),
        # Rewrite operation
        union(src_ary.getPort(1)).with_(
            Nb_Array_Neg(io, ary, itemty)
        ),
        # Return type is ary
        set_(TypeVar(src_ary.getPort(1)).getType()).to(ty),
    )


@function
def Nb_Array_Neg(
    io: Term, ary: Term, dtype: Type
) -> Term: ...


class NbOp_Array_Neg_Unary(NbOp_Base):
    io: SExpr
    ary: SExpr
    attr: SExpr


# ### Extend egraph extraction
   
class ExtendEGraphToRVSDG(_ExtendEGraphToRVSDG):
    def handle_Term(self, op: str, children: dict | list, grm: Grammar):
        match op, children:
            case "Nb_Array_Neg", {
                "io": io,
                "ary": ary,
                "dtype": dtype,
            }:
                return grm.write(
                    NbOp_Array_Neg_Unary(
                        io=io,
                        ary=ary,
                        attr=grm.write(rg.Attrs(dtype)),
                    )
                )
        return super().handle_Term(op, children, grm)


class Backend(_Backend):
    def lower_type(self, ty: NbOp_Type):
        match ty:
            case NbOp_ArrayType(
                dtype=dtype,
                ndim=int(ndim),
                datalayout=str(datalayout),
                shape=shape,
            ):
                mlir_dtype = self.lower_type(dtype)
                with self.loc:
                    memref_ty=ir.MemRefType.get([10]*ndim, mlir_dtype)
                return memref_ty
        return super().lower_type(ty)


    def lower_expr(self, expr: SExpr, state: LowerStates):
        match expr:
            case NbOp_Array_Neg_Unary(
                io=io, ary=ary, attr=attr
            ):
                io = yield io
                ary = yield ary
                memref_ty = ir.MemRefType.get([10, 10], self.f64)
                ufun = create_unary_ufunc(arith.negf, memref_ty, self.module_body, self.f64)
                res = memref.AllocOp(memref_ty, [], [])
                func.CallOp(ufun, [ary, res])
                return res

        return (yield from super().lower_expr(expr, state))

# ### `ctypes` definition for Array

class CtypeInt64Array2D(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p), ("shape", (ctypes.c_uint64 * 2))]


array_float64_2d, array_infos = array_desc_rules(
    "array_float64_2d", shape=("m", "n"), dtype=TypeFloat64, layout="c"
)
ufunc_counter=0
def create_unary_ufunc(operation, memref_ty, module_body, f64):
    global ufunc_counter
    ufunc_counter += 1
    with module_body:
        ufun = func.FuncOp(f"ufunc_{ufunc_counter}", ((memref_ty, memref_ty), ()))
        ufun.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        uconst_block = ufun.add_entry_block()
        uconstant_entry = ir.InsertionPoint(uconst_block)

        with uconstant_entry:
            array_1, res = ufun.arguments

            indexing_maps = ir.ArrayAttr.get([
                ir.AffineMapAttr.get(ir.AffineMap.get(2, 0, [
                    ir.AffineExpr.get_dim(0),
                    ir.AffineExpr.get_dim(1),
                ])),
                ir.AffineMapAttr.get(ir.AffineMap.get(2, 0, [
                    ir.AffineExpr.get_dim(0),
                    ir.AffineExpr.get_dim(1),
                ])),
            ])
            iterators = ir.ArrayAttr.get([
                ir.Attribute.parse(f"#linalg.iterator_type<parallel>"),
                ir.Attribute.parse(f"#linalg.iterator_type<parallel>"),
            ])
            matmul = linalg.GenericOp(
                result_tensors=[],
                inputs=[array_1],
                outputs=[res],
                indexing_maps=indexing_maps,
                iterator_types=iterators
            )

            body = matmul.regions[0].blocks.append(f64, f64)
            with ir.InsertionPoint(body):
                a, b = body.arguments
                m = operation(a)
                linalg.YieldOp([m])
            func.ReturnOp([])

    return ufun

def create_binary_ufunc(operation, memref_ty, module_body, f64):
    global ufunc_counter
    ufunc_counter += 1
    with module_body:
        ufun = func.FuncOp(f"ufunc_{ufunc_counter}", ((memref_ty, memref_ty, memref_ty), ()))
        ufun.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        uconst_block = ufun.add_entry_block()
        uconstant_entry = ir.InsertionPoint(uconst_block)

        with uconstant_entry:
            array_1, array_2, res = ufun.arguments

            indexing_maps = ir.ArrayAttr.get([
                ir.AffineMapAttr.get(ir.AffineMap.get(2, 0, [
                    ir.AffineExpr.get_dim(0),
                    ir.AffineExpr.get_dim(1),
                ])),
                ir.AffineMapAttr.get(ir.AffineMap.get(2, 0, [
                    ir.AffineExpr.get_dim(0),
                    ir.AffineExpr.get_dim(1),
                ])),
                ir.AffineMapAttr.get(ir.AffineMap.get(2, 0, [
                    ir.AffineExpr.get_dim(0),
                    ir.AffineExpr.get_dim(1),
                ])),
            ])
            iterators = ir.ArrayAttr.get([
                ir.Attribute.parse(f"#linalg.iterator_type<parallel>"),
                ir.Attribute.parse(f"#linalg.iterator_type<parallel>"),
                ir.Attribute.parse(f"#linalg.iterator_type<parallel>"),
            ])
            matmul = linalg.GenericOp(
                result_tensors=[],
                inputs=[array_1, array_2],
                outputs=[res],
                indexing_maps=indexing_maps,
                iterator_types=iterators
            )

            body = matmul.regions[0].blocks.append(f64, f64)
            with ir.InsertionPoint(body):
                a, b = body.arguments
                m = operation(a, b)
                linalg.YieldOp([m])
            func.ReturnOp([])
    
    return ufun

def example_1(a):
    c = - a
    return c

if __name__ == "__main__":
    # compile
    jt = compiler_pipeline(
        example_1,
        argtypes=(array_2d_symbolic,),
        ruleset=(
            base_ruleset
            | setup_argtypes(array_float64_2d.toType())
            | ruleset(*array_infos)
            | ruleset_typeinfer_array_neg
        ),
        verbose=True,
        converter_class=ExtendEGraphToRVSDG,
        cost_model=MyCostModel(),
        backend=Backend(),
    )
    # create array
    ary = np.arange(100, dtype=np.int64).reshape(10, 10)

    got = jt(ary)
    print("GOT", got)

    ary_exp = np.arange(100, dtype=np.int64).reshape(10, 10)
    res_exp = np.zeros(100, dtype=np.int64).reshape(10, 10)

    # compare the result
    expect = example_1(ary_exp, res_exp)
    assert res == res_exp
