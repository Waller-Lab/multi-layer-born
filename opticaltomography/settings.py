"""
variables setting
"""
import arrayfire as af
bit = 32

np_float_datatype   = "float32" if bit == 32 else "float64"
np_complex_datatype = "complex64" if bit == 32 else "complex128"
af_float_datatype   = af.Dtype.f32 if bit == 32 else af.Dtype.f64
af_complex_datatype = af.Dtype.c32 if bit == 32 else af.Dtype.c64