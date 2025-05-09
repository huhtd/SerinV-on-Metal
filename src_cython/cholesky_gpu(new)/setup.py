from setuptools import setup, Extension
from Cython.Build import cythonize
from distutils.command.clean import clean as _clean  
import pathlib                                        
import numpy

ext = Extension(
    "metal_cholesky",
    sources=["metal_cholesky.pyx", "mps_cholesky.m"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-fobjc-arc"],
    extra_link_args=[
        "-framework", "Metal",
        "-framework", "MetalPerformanceShaders",
    ],
    language="c",
)

# --- clean: remove generated metal_cholesky.c ------------
class clean(_clean):
    def run(self):
        _clean.run(self)
        p = pathlib.Path("metal_cholesky.c")
        if p.exists():
            print(f"removing {p}")
            p.unlink()

setup(
    name="metal_cholesky",
    ext_modules=cythonize([ext], language_level="3"),
    cmdclass={"clean": clean},         
    zip_safe=False,
)
