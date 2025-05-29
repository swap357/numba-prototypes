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

context = ir.Context()
loc = ir.Location.unknown(context=context)
module = ir.Module.create(loc=loc)

f64 = ir.F64Type.get(context=context)
index_type = ir.IndexType.get(context=context)

with context, loc:
    memref_ty = ir.MemRefType.get([10, 10], f64)

module_body = ir.InsertionPoint(module.body)

input_types = (memref_ty, memref_ty, memref_ty)
output_types = ()


with context, loc, module_body:
    fun = func.FuncOp("func", (input_types, output_types))
    fun.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    const_block = fun.add_entry_block()
    constant_entry = ir.InsertionPoint(const_block)

    with constant_entry:
        arg_1, arg_2, res = fun.arguments
        res_type = ir.RankedTensorType.get([10, 10], f64)

        indexing_maps = ir.ArrayAttr.get([
            ir.AffineMapAttr.get(ir.AffineMap.get(3, 0, [
                ir.AffineExpr.get_dim(0),
                ir.AffineExpr.get_dim(2),
            ])),
            ir.AffineMapAttr.get(ir.AffineMap.get(3, 0, [
                ir.AffineExpr.get_dim(2),
                ir.AffineExpr.get_dim(1),
            ])),
            ir.AffineMapAttr.get(ir.AffineMap.get(3, 0, [
                ir.AffineExpr.get_dim(0),
                ir.AffineExpr.get_dim(1),
            ])),
        ])
        iterators = ir.ArrayAttr.get([
            ir.Attribute.parse(f"#linalg.iterator_type<parallel>"),
            ir.Attribute.parse(f"#linalg.iterator_type<parallel>"),
            ir.Attribute.parse(f"#linalg.iterator_type<reduction>"),
        ])
        matmul = linalg.GenericOp(
            result_tensors=[],
            inputs=[arg_1, arg_2],
            outputs=[res],
            indexing_maps=indexing_maps,
            iterator_types=iterators
        )

        body = matmul.regions[0].blocks.append(f64, f64, f64)
        with ir.InsertionPoint(body):
            a, b, r = body.arguments
            m = arith.mulf(a, b)
            s = arith.addf(r, m)
            linalg.YieldOp([s])

        func.ReturnOp([])

_DEBUG=False
_DEBUG=True

if _DEBUG:
    context.enable_multithreading(False)
pass_man = passmanager.PassManager(context=context)
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

if _DEBUG:
    module.dump()

cuda_libs = ("mlir_cuda_runtime", "mlir_c_runner_utils", "mlir_runner_utils")
cuda_shared_libs = [find_library(x) for x in cuda_libs]

engine = execution_engine.ExecutionEngine(module,
                                          opt_level=3,
                                          shared_libs=cuda_shared_libs)

array_1 = np.arange(100, dtype=np.float64).reshape((10,10))
array_2 = np.arange(100, dtype=np.float64).reshape((10,10))
res = np.zeros(100, dtype=np.float64).reshape((10,10))

ctlie = namedtuple("ctypes_lie", "data data_as shape")

def np_arr_to_np_duck_device_arr(arr):
    da = cuda.to_device(arr)
    da.ctypes = ctlie(da.__cuda_array_interface__["data"][0],
                lambda x: ctypes.cast(da.ctypes.data, x),
                da.__cuda_array_interface__["shape"],)
    da.itemsize = arr.itemsize
    return da

da_array_1 = np_arr_to_np_duck_device_arr(array_1)
da_array_2 = np_arr_to_np_duck_device_arr(array_2)
da_res = np_arr_to_np_duck_device_arr(res)


array1_as_memref = runtime.get_ranked_memref_descriptor(da_array_1)
array2_as_memref = runtime.get_ranked_memref_descriptor(da_array_2)
res_as_memref = runtime.get_ranked_memref_descriptor(da_res)

engine.invoke("func", ctypes.pointer(ctypes.pointer(array1_as_memref)),
                      ctypes.pointer(ctypes.pointer(array2_as_memref)),
                      ctypes.pointer(ctypes.pointer(res_as_memref)))


print("Done execution")
result = da_res.copy_to_host()
np.testing.assert_allclose(np.dot(array_1, array_2), result)
print("Results match expected")
