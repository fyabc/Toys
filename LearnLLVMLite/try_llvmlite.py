# -*- coding: utf-8 -*-

from ctypes import CFUNCTYPE, c_int

import llvmlite.ir as ll
import llvmlite.binding as llvm

__author__ = 'fyabc'

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

# Create a new module with a function implementing this:
#
# int sum(int a, int b) {
#   return a + b;
# }
module = ll.Module()
func_ty = ll.FunctionType(ll.IntType(32), [ll.IntType(32), ll.IntType(32)])
func = ll.Function(module, func_ty, name='sum')

func.args[0].name = 'a'
func.args[1].name = 'b'

bb_entry = func.append_basic_block('entry')
ir_builder = ll.IRBuilder(bb_entry)
tmp = ir_builder.add(func.args[0], func.args[1])
ret = ir_builder.ret(tmp)

print('=== LLVM IR')
print(module)

# Convert textual LLVM IR into in-memory representation.
llvm_module = llvm.parse_assembly(str(module))

tm = llvm.Target.from_default_triple().create_target_machine()

# Compile the module to machine code using MCJIT
with llvm.create_mcjit_compiler(llvm_module, tm) as ee:
    ee.finalize_object()
    print('=== Assembly')
    print(tm.emit_assembly(llvm_module))
    # print('=== Object')
    # print(tm.emit_object(llvm_module))

    # Obtain a pointer to the compiled 'sum' - it's the address of its JITed
    # code in memory.
    # cfptr = ee.get_pointer_to_function(llvm_module.get_function('sum'))
    cfptr = ee.get_function_address('sum')

    # To convert an address to an actual callable thing we have to use
    # CFUNCTYPE, and specify the arguments & return type.
    cfunc = CFUNCTYPE(c_int, c_int, c_int)(cfptr)

    # Now 'cfunc' is an actual callable we can invoke
    res = cfunc(17, 42)
    print('The result is', res)
