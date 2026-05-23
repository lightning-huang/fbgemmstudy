import fbgemm_gpu.sparse_ops as sparse

# Inspect the abstract functions
print("=== int_nbit_split_embedding_codegen_lookup_function_meta ===")
meta = sparse.int_nbit_split_embedding_codegen_lookup_function_meta
import inspect
sig = inspect.signature(meta)
print("signature:", sig)
print("module:", meta.__module__)

print()
print("=== SparseType ===")
st = sparse.SparseType
for x in st:
    print(" ", x.name, "=", x.value)

# Try to see if there's a TBE class in sparse_ops
print()
print("All sparse_ops functions:")
for name in dir(sparse):
    if not name.startswith("_") and callable(getattr(sparse, name)):
        obj = getattr(sparse, name)
        if not isinstance(obj, type):
            print("  ", name)