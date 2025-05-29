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

# # Ch 8. GPU offloading for MLIR ufunc operations
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

import mlir.dialects.arith as arith
import mlir.dialects.affine as affine
import mlir.dialects.memref as memref
import mlir.dialects.scf as scf
import mlir.dialects.func as func
import mlir.dialects.linalg as linalg
import mlir.dialects.bufferization as bufferization
import mlir.execution_engine as execution_engine
import mlir.ir as ir
import mlir.runtime as runtime
import mlir.passmanager as passmanager
import ctypes
from ctypes.util import find_library
import numpy as np
from numba import cuda
from collections import namedtuple

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
_DEBUG = True

class GPUBackend(_Backend):
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

    def run_passes(self, module):
        module.dump()
        pass_man = passmanager.PassManager(context=module.context)        

        if _DEBUG:
            module.context.enable_multithreading(False)
        if _DEBUG:
            pass_man.enable_ir_printing()   

        pass_man.add("convert-linalg-to-affine-loops")
        pass_man.add("affine-loop-fusion")
        pass_man.add("func.func(affine-parallelize)")
        pass_man.add("builtin.module(func.func(gpu-map-parallel-loops,convert-parallel-loops-to-gpu))")
        pass_man.add("lower-affine")
        pass_man.add("scf-parallel-loop-fusion")
        pass_man.add('func.func(gpu-map-parallel-loops,convert-parallel-loops-to-gpu)')
        pass_man.add("gpu-kernel-outlining")
        pass_man.add('gpu-lower-to-nvvm-pipeline{cubin-format="fatbin"}')
        pass_man.add("convert-scf-to-cf")
        pass_man.add("finalize-memref-to-llvm")
        pass_man.add("convert-func-to-llvm")
        pass_man.add("convert-index-to-llvm")
        pass_man.add("convert-bufferization-to-memref")
        pass_man.add("reconcile-unrealized-casts")
        pass_man.add("func.func(llvm-request-c-wrappers)")
        pass_man.enable_verifier(True)
        pass_man.run(module.operation)
        # Output LLVM-dialect MLIR
        module.dump()
        return module

    @classmethod
    def get_exec_ptr(cls, mlir_ty, val, out_val=False):
        if isinstance(mlir_ty, ir.IntegerType):
            val = 0 if val is None else val
            ptr = ctypes.pointer(ctypes.c_int64(val))
        elif isinstance(mlir_ty, ir.F32Type):
            val = 0.0 if val is None else val
            ptr = ctypes.pointer(ctypes.c_float(val))
        elif isinstance(mlir_ty, ir.F64Type):
            val = 0.0 if val is None else val
            ptr = ctypes.pointer(ctypes.c_double(val))
        elif isinstance(mlir_ty, ir.MemRefType):
            # TODO: Remove this hardcoded shape value
            val = np.zeros((10, 10)) if val is None else val
            ptr = ctypes.pointer(ctypes.pointer(runtime.get_ranked_memref_descriptor(cls.np_arr_to_np_duck_device_arr(val))))

        if out_val:
            return ptr, val
        else:
            return ptr
    
    @classmethod
    def np_arr_to_np_duck_device_arr(cls, arr):
        da = cuda.to_device(arr)
        ctlie = namedtuple("ctypes_lie", "data data_as shape")
        da.ctypes = ctlie(da.__cuda_array_interface__["data"][0],
                    lambda x: ctypes.cast(da.ctypes.data, x),
                    da.__cuda_array_interface__["shape"],)
        da.itemsize = arr.itemsize
        return da

compiler.set_backend(GPUBackend())
compiler.set_converter_class(ConditionalExtendGraphtoRVSDG)
compiler.set_cost_model(MyCostModel())

# Decorator function for vecotrization.
def gpu_vectorize(input_types, shape=None, ndim=None):
    num_inputs = len(input_types)

    def to_input_dtypes(input_tys):
        res = []
        for ty in input_tys:
            if ty == Float64:
                res.append(TypeFloat64)
        return tuple(res)

    def wrapper(inner_func):
        nonlocal ndim
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

            if ndim is not None:
                memref_ty = ir.MemRefType.get([ir.ShapedType.get_dynamic_size()] * ndim, f64)
            elif shape is not None:
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
                ] * (num_inputs + 1))
                matmul = linalg.GenericOp(
                    result_tensors=[],
                    inputs=arys,
                    outputs=[res],
                    indexing_maps=indexing_maps,
                    iterator_types=iterators
                )
                # Within the affine loop body make calls to the inner function.
                body = matmul.regions[0].blocks.append(*([f64] * num_inputs))
                with ir.InsertionPoint(body):
                    m = func.CallOp([f64], "func", [*body.arguments])
                    linalg.YieldOp([m])
                func.ReturnOp([])

        compiler.run_backend_passes(llmod)
        cuda_libs = ("mlir_cuda_runtime", "mlir_c_runner_utils", "mlir_runner_utils")
        cuda_shared_libs = [find_library(x) for x in cuda_libs]
        jit_func = compiler.compile_module_(llmod, [memref_ty] * num_inputs, (memref_ty,), "ufunc",
                                            exec_engine=execution_engine.ExecutionEngine(llmod, opt_level=3, shared_libs=cuda_shared_libs))
        return jit_func

    return wrapper


@gpu_vectorize(input_types=[Float64, Float64, Float64], ndim=2)
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

