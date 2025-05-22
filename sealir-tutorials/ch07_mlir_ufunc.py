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
from ch04_1_typeinfer_ifelse import TypeFloat64
from ch04_2_typeinfer_loops import (
    MyCostModel,
    base_ruleset,
    compiler_pipeline,
    setup_argtypes,
)
from ch05_typeinfer_array import NbOp_ArrayType, NbOp_ArrayDimSymbolic, Type
from ch06_mlir_backend import ConditionalExtendGraphtoRVSDG as _ExtendEGraphToRVSDG, Backend as _Backend, NbOp_Type

import mlir.dialects.arith as arith
import mlir.dialects.math as math
import mlir.dialects.memref as memref
import mlir.dialects.linalg as linalg
import mlir.dialects.cf as cf
import mlir.dialects.func as func
import mlir.dialects.scf as scf
import mlir.execution_engine as execution_engine
import mlir.ir as ir
import mlir.runtime as runtime
import mlir.passmanager as passmanager

Float64 = NbOp_Type("Float64")
TypeFloat64 = Type.simple("Float64")

array_2d_symbolic = NbOp_ArrayType(
    dtype=Float64,
    ndim=2,
    datalayout="c_contiguous",
    shape=(NbOp_ArrayDimSymbolic("m"), NbOp_ArrayDimSymbolic("n")),
)


# @ruleset
# def ruleset_typeinfer_array_neg(
#     src_ary: Term,
#     io: Term,
#     ary: Term,
#     ty: Type,
#     ary_uid: String,
#     arydesc: ArrayDesc,
#     itemty: Type,
# ):
#     yield rule(
#         # Implement getitem(int)->scalar
#         src_ary == Py_NegIO(io, ary),
#         # ary is array type
#         ty == TypeVar(ary).getType(),
#         ty == arydesc.toType(),
#         # get item type
#         itemty == arydesc.dtype,
#     ).then(
#         # shortcut IO
#         union(src_ary.getPort(0)).with_(io),
#         # Rewrite operation
#         union(src_ary.getPort(1)).with_(
#             Nb_Array_Neg(io, ary, itemty)
#         ),
#         # Return type is ary
#         set_(TypeVar(src_ary.getPort(1)).getType()).to(ty),
#     )


# @ruleset
# def ruleset_typeinfer_array_add(
#     src_ary: Term,
#     io: Term,
#     ary_1: Term,
#     ary_2: Term,
#     ty: Type,
#     ary_uid: String,
#     arydesc: ArrayDesc,
#     itemty: Type,
# ):
#     yield rule(
#         # Implement getitem(int)->scalar
#         src_ary == Py_AddIO(io, ary_1, ary_2),
#         # ary is array type
#         ty == TypeVar(ary).getType(),
#         ty == arydesc.toType(),
#         # get item type
#         itemty == arydesc.dtype,
#     ).then(
#         # shortcut IO
#         union(src_ary.getPort(0)).with_(io),
#         # Rewrite operation
#         union(src_ary.getPort(1)).with_(
#             Nb_Array_Add(io, ary, itemty)
#         ),
#         # Return type is ary
#         set_(TypeVar(src_ary.getPort(1)).getType()).to(ty),
#     )

# @function
# def Nb_Array_Neg(
#     io: Term, ary: Term, dtype: Type
# ) -> Term: ...


# class NbOp_Array_Neg_Unary(NbOp_Base):
#     io: SExpr
#     ary: SExpr
#     attr: SExpr


# @function
# def Nb_Array_Add(
#     io: Term, ary_1: Term, ary_2: Term, dtype: Type
# ) -> Term: ...


# class NbOp_Array_Add_Binary(NbOp_Base):
#     io: SExpr
#     ary_1: SExpr
#     ary_2: SExpr
#     attr: SExpr



# ### Extend egraph extraction
   
class ExtendEGraphToRVSDG(_ExtendEGraphToRVSDG):
    pass
#     def handle_Term(self, op: str, children: dict | list, grm: Grammar):
#         match op, children:
#             case "Nb_Array_Neg", {
#                 "io": io,
#                 "ary": ary,
#                 "dtype": dtype,
#             }:
#                 return grm.write(
#                     NbOp_Array_Neg_Unary(
#                         io=io,
#                         ary=ary,
#                         attr=grm.write(rg.Attrs(dtype)),
#                     )
#                 )
#         return super().handle_Term(op, children, grm)


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


    # def lower_expr(self, expr: SExpr, state: LowerStates):
    #     match expr:
    #         case NbOp_Array_Neg_Unary(
    #             io=io, ary=ary, attr=attr
    #         ):
    #             io = yield io
    #             ary = yield ary
    #             memref_ty = ir.MemRefType.get([10, 10], self.f64)
    #             ufun = create_unary_ufunc(arith.negf, memref_ty, self.module_body, self.f64)
    #             res = memref.AllocOp(memref_ty, [], [])
    #             func.CallOp(ufun, [ary, res])
    #             return res

    #     return (yield from super().lower_expr(expr, state))

# ufunc_counter=0
# def create_unary_ufunc(operation, memref_ty, module_body, f64):
#     global ufunc_counter
#     ufunc_counter += 1
#     with module_body:
#         ufun = func.FuncOp(f"ufunc_{ufunc_counter}", ((memref_ty, memref_ty), ()))
#         ufun.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
#         uconst_block = ufun.add_entry_block()
#         uconstant_entry = ir.InsertionPoint(uconst_block)

#         with uconstant_entry:
#             array_1, res = ufun.arguments

#             indexing_maps = ir.ArrayAttr.get([
#                 ir.AffineMapAttr.get(ir.AffineMap.get(2, 0, [
#                     ir.AffineExpr.get_dim(0),
#                     ir.AffineExpr.get_dim(1),
#                 ])),
#                 ir.AffineMapAttr.get(ir.AffineMap.get(2, 0, [
#                     ir.AffineExpr.get_dim(0),
#                     ir.AffineExpr.get_dim(1),
#                 ])),
#             ])
#             iterators = ir.ArrayAttr.get([
#                 ir.Attribute.parse(f"#linalg.iterator_type<parallel>"),
#                 ir.Attribute.parse(f"#linalg.iterator_type<parallel>"),
#             ])
#             matmul = linalg.GenericOp(
#                 result_tensors=[],
#                 inputs=[array_1],
#                 outputs=[res],
#                 indexing_maps=indexing_maps,
#                 iterator_types=iterators
#             )

#             body = matmul.regions[0].blocks.append(f64, f64)
#             with ir.InsertionPoint(body):
#                 a, b = body.arguments
#                 m = operation(a)
#                 linalg.YieldOp([m])
#             func.ReturnOp([])

#     return ufun

# def create_binary_ufunc(operation, memref_ty, module_body, f64):
#     global ufunc_counter
#     ufunc_counter += 1
#     with module_body:
#         ufun = func.FuncOp(f"ufunc_{ufunc_counter}", ((memref_ty, memref_ty, memref_ty), ()))
#         ufun.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
#         uconst_block = ufun.add_entry_block()
#         uconstant_entry = ir.InsertionPoint(uconst_block)

#         with uconstant_entry:
#             array_1, array_2, res = ufun.arguments

#             indexing_maps = ir.ArrayAttr.get([
#                 ir.AffineMapAttr.get(ir.AffineMap.get(2, 0, [
#                     ir.AffineExpr.get_dim(0),
#                     ir.AffineExpr.get_dim(1),
#                 ])),
#                 ir.AffineMapAttr.get(ir.AffineMap.get(2, 0, [
#                     ir.AffineExpr.get_dim(0),
#                     ir.AffineExpr.get_dim(1),
#                 ])),
#                 ir.AffineMapAttr.get(ir.AffineMap.get(2, 0, [
#                     ir.AffineExpr.get_dim(0),
#                     ir.AffineExpr.get_dim(1),
#                 ])),
#             ])
#             iterators = ir.ArrayAttr.get([
#                 ir.Attribute.parse(f"#linalg.iterator_type<parallel>"),
#                 ir.Attribute.parse(f"#linalg.iterator_type<parallel>"),
#                 ir.Attribute.parse(f"#linalg.iterator_type<parallel>"),
#             ])
#             matmul = linalg.GenericOp(
#                 result_tensors=[],
#                 inputs=[array_1, array_2],
#                 outputs=[res],
#                 indexing_maps=indexing_maps,
#                 iterator_types=iterators
#             )

#             body = matmul.regions[0].blocks.append(f64, f64)
#             with ir.InsertionPoint(body):
#                 a, b = body.arguments
#                 m = operation(a, b)
#                 linalg.YieldOp([m])
#             func.ReturnOp([])
    
#     return ufun

Float64 = NbOp_Type("Float64")

def ufunc_vectorize(input_types, output_types, shape=None, ndim=None):

    def to_input_dtypes(input_tys):
        res = []
        for ty in input_tys:
            if ty == Float64:
                res.append(TypeFloat64)
        return tuple(res)

    def wrapper(inner_func):
        # compile
        llmod = compiler_pipeline(
            inner_func,
            argtypes=input_types,
            ruleset=(
                base_ruleset
                | setup_argtypes(*to_input_dtypes(input_types))
            ),
            verbose=True,
            converter_class=ExtendEGraphToRVSDG,
            cost_model=MyCostModel(),
            backend=Backend(),
            return_module=True
        )
        module_body = ir.InsertionPoint(llmod.body)
        context = llmod.context
        loc = ir.Location.unknown(context=context)

        f64 = ir.F64Type.get(context=context)
        index_type = ir.IndexType.get(context=context)

        with context, loc:
            # TODO: Wire shapes into this
            memref_ty_undef = ir.MemRefType.get([ir.ShapedType.get_dynamic_size(), ir.ShapedType.get_dynamic_size()], f64)
            memref_ty_1 = ir.MemRefType.get([10, 10], f64)
            memref_ty_2 = ir.MemRefType.get([10, 20], f64)

        input_typ_outer = (memref_ty_undef, memref_ty_undef, memref_ty_undef)
        output_typ_outer = ()
        with context, loc, module_body:
            fun = func.FuncOp("ufunc", (input_typ_outer, output_typ_outer))
            fun.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            const_block = fun.add_entry_block()
            constant_entry = ir.InsertionPoint(const_block)
            
            with constant_entry:
                array_1, array_2, res = fun.arguments

                # TODO: Need proper dimensional wiring
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
                    m = func.CallOp([f64], "func", [a, b])
                    linalg.YieldOp([m])
                func.ReturnOp([])

        llmod.dump()
        pass_man = passmanager.PassManager(context=context)

        # pass_man.add("lower-affine")
        # pass_man.add("convert-tensor-to-linalg")
        # pass_man.add("convert-linalg-to-affine-loops")
        # pass_man.add("affine-loop-fusion")
        # pass_man.add("affine-parallelize")

        pass_man.add("convert-linalg-to-loops")
        pass_man.add("convert-scf-to-cf")
        pass_man.add("finalize-memref-to-llvm")
        pass_man.add("convert-func-to-llvm")
        pass_man.add("convert-index-to-llvm")
        pass_man.add("reconcile-unrealized-casts")
        pass_man.enable_verifier(True)
        pass_man.run(llmod.operation)
        llmod.dump()

        engine = execution_engine.ExecutionEngine(llmod)

        def inner_wrapper(*args):
            # TODO: Check args properly with input shape and declare resulting array accordingly
            res_array = np.zeros_like(args[0])
            engine_args = [ctypes.pointer(ctypes.pointer(runtime.get_ranked_memref_descriptor(arg))) for arg in (*args, res_array)]
            engine.invoke("ufunc", *engine_args)
            return res_array
        
        return inner_wrapper

    return wrapper


@ufunc_vectorize(input_types=[Float64, Float64], output_types=[Float64], shape=(10, 10))
def foo(a, b):
    x = a + 1.0
    y = b - 2.0
    return x + y

if __name__ == "__main__":
    # create array
    ary = np.arange(100, dtype=np.float64).reshape(10, 10)
    ary_2 = np.arange(100, dtype=np.float64).reshape(10, 10)

    got = foo(ary, ary_2)
    print("Got", got)

