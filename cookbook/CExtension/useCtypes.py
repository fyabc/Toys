#! /usr/bin/python

import ctypes
import os

__author__ = 'fyabc'

# Try to locate the shared library
_file = 'my_utils.dll'
_path = os.path.join(*(os.path.split(__file__)[:-1] + (_file,)))
_module = ctypes.cdll.LoadLibrary(_path)


# void myPrint(int)
myPrint = _module.myPrint
myPrint.argtypes = (ctypes.c_int,)
myPrint.restype = None


# int gcd(int, int)
gcd = _module.gcd
gcd.argtypes = (ctypes.c_int, ctypes.c_int)
gcd.restype = ctypes.c_int


# int inMandel(double, double, int)
inMandel = _module.inMandel
inMandel.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_int)
inMandel.restype = ctypes.c_int


# int divMod(int, int, int*)
_divMod = _module.divMod
_divMod.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
_divMod.restype = ctypes.c_int


def divMod(x, y):
    r = ctypes.c_int()
    q = _divMod(x, y, r)

    return q, r


# void avg(double*, int)
# Define a special type for 'double *' argument
class DoubleArrayType:
    def fromParam(self, param):
        typename = type(param).__name__

        if hasattr(self, 'from_' + typename):
            return getattr(self, 'from_' + typename)(param)
        elif isinstance(param, ctypes.Array):
            return param
        else:
            raise TypeError('Cannot convert %s to a double array' % typename)

    # Cast from array.array objects


def test():
    print(myPrint(4))


if __name__ == '__main__':
    test()
