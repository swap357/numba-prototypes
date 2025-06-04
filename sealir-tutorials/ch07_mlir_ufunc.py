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

import numpy as np
from ch04_1_typeinfer_ifelse import TypeFloat64
from ch04_2_typeinfer_loops import (
    MyCostModel,
    base_ruleset,
    compiler,
    setup_argtypes,
)
from ch05_typeinfer_array import NbOp_ArrayType, NbOp_ArrayDimSymbolic, Type
from ch06_mlir_backend import ConditionalExtendGraphtoRVSDG, Backend as _Backend, NbOp_Type

import mlir.dialects.linalg as linalg
import mlir.dialects.func as func
import mlir.ir as ir

# Type declaration for array elements
Float64 = NbOp_Type("Float64")
TypeFloat64 = Type.simple("Float64")

# Define an array using the Float64 dtypes
# and symbolic dimensions (m, n)
array_2d_symbolic = NbOp_ArrayType(
    dtype=Float64,
    ndim=2,
    datalayout="c_contiguous",
    shape=(NbOp_ArrayDimSymbolic("m"), NbOp_ArrayDimSymbolic("n")),
)

class Backend(_Backend):
    # Lower symbolic array to respective memref.
    # Note: This is not used within ufunc builder,
    # since it has explicit declaration of the respective
    # MLIR memrefs. 
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
                    memref_ty=ir.MemRefType.get(shape, mlir_dtype)
                return memref_ty
        return super().lower_type(ty)

compiler.set_backend(Backend())
compiler.set_converter_class(ConditionalExtendGraphtoRVSDG)
compiler.set_cost_model(MyCostModel())

# Decorator function for vecotrization.
def ufunc_vectorize(input_types, shape):
    num_inputs = len(input_types)

    def to_input_dtypes(input_tys):
        res = []
        for ty in input_tys:
            if ty == Float64:
                res.append(TypeFloat64)
        return tuple(res)

    def wrapper(inner_func):
        # Compile the inner function and get the IR as a module.
        llmod, func_egraph = compiler.lower_py_fn(
            inner_func,
            argtypes=input_types,
            ruleset=(
                base_ruleset
                | setup_argtypes(*to_input_dtypes(input_types))
            ),
        )

        # Now within the module declare a seperate function named 
        # 'ufunc' which acts as a wrapper around the innner 'func'
        with llmod.context, ir.Location.unknown(context=llmod.context), ir.InsertionPoint(llmod.body):
            f64 = ir.F64Type.get()

            ndim = len(shape)
            memref_ty = ir.MemRefType.get(shape, f64)

            # The function 'ufunc' has N + 1 number of arguments 
            # (where N is the nuber of arguments for the original function)
            # The extra argument is an explicitly declared resulting array.
            input_typ_outer = (memref_ty,) * (num_inputs + 1)

            fun = func.FuncOp("ufunc", (input_typ_outer, ()))
            fun.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            const_block = fun.add_entry_block()
            constant_entry = ir.InsertionPoint(const_block)

            # Within this function we declare the symbolic representation of
            # input and output arrays of appropriate shapes using memrefs.
            with constant_entry:
                arys = fun.arguments[:-1]
                res = fun.arguments[-1]

                # Affine map declaration
                indexing_maps = ir.ArrayAttr.get([
                    ir.AffineMapAttr.get(ir.AffineMap.get(ndim, 0, [
                        ir.AffineExpr.get_dim(i) for i in range(ndim)
                    ])),
                ] * (num_inputs + 1))
                iterators = ir.ArrayAttr.get([
                    ir.Attribute.parse(f"#linalg.iterator_type<parallel>")
                ] * (1 + 1))
                matmul = linalg.GenericOp(
                    result_tensors=[],
                    inputs=arys,
                    outputs=[res],
                    indexing_maps=indexing_maps,
                    iterator_types=iterators
                )
                # Within the affine loop body make calls to the inner function.
                body = matmul.regions[0].blocks.append(*([f64] * (num_inputs + 1)))
                with ir.InsertionPoint(body):
                    m = func.CallOp([f64], "func", [*body.arguments[:-1]])
                    linalg.YieldOp([m])
                func.ReturnOp([])

        compiler.run_backend_passes(llmod)

        jit_func = compiler.compile_module_(llmod, [memref_ty] * num_inputs, (memref_ty,), "ufunc")
        return jit_func

    return wrapper


@ufunc_vectorize(input_types=[Float64, Float64, Float64], shape=(10, 10))
def foo(a, b, c):
    x = a + 1.0
    y = b - 2.0
    z = c + 3.0
    return x + y + z

if __name__ == "__main__":
    # Create NumPy arrays 
    ary = np.arange(100, dtype=np.float64).reshape(10, 10)
    ary_2 = np.arange(100, dtype=np.float64).reshape(10, 10)
    ary_3 = np.arange(100, dtype=np.float64).reshape(10, 10)

    got = foo(ary, ary_2, ary_3)
    print("Got", got)

