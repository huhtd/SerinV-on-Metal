from setuptools import setup, Extension
from Cython.Build import cythonize
from distutils.command.clean import clean as _clean  
import pathlib                                        
import numpy
import shutil  # For removing directories

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

# --- clean: remove generated files including the build folder ------------
class clean(_clean):
    def run(self):
        _clean.run(self)
        
        # Remove the metal_cholesky.c file
        p = pathlib.Path("metal_cholesky.c")
        if p.exists():
            print(f"removing {p}")
            p.unlink()

        # Remove the build folder
        build_dir = pathlib.Path("build")
        if build_dir.exists() and build_dir.is_dir():
            print(f"removing {build_dir}")
            shutil.rmtree(build_dir)

        # Remove any .so file in the current directory
        for so_file in pathlib.Path(".").glob("*.so"):
            if so_file.exists():
                print(f"removing {so_file}")
                so_file.unlink()

setup(
    name="metal_cholesky",
    ext_modules=cythonize([ext], language_level="3"),
    cmdclass={"clean": clean},         
    zip_safe=False,
)
