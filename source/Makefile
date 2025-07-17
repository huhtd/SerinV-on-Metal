CXX = clang++
CXXFLAGS = -std=c++17 -fobjc-arc
LDFLAGS = -dynamiclib
MPS_FRAMEWORKS = -framework Foundation -framework Metal -framework MetalPerformanceShaders
ACC_FRAMEWORK = -framework Accelerate

# Files
MUL_SRC      = mps_mul.mm
MPS_SRC      = mps_cholesky.mm
BLAS_SRC     = blas_cholesky.cpp
BLAS_MUL_SRC = blas_mul.cpp

# Outputs
MUL_LIB      = libmpsmul.dylib
MPS_LIB      = libmpscholesky.dylib
BLAS_LIB     = libblascholesky.dylib
BLAS_MUL_LIB = libblasmul.dylib

.PHONY: all clean

all: $(MUL_LIB) $(MPS_LIB) $(BLAS_LIB) $(BLAS_MUL_LIB)

$(MUL_LIB): $(MUL_SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(MPS_FRAMEWORKS) $< -o $@

$(MPS_LIB): $(MPS_SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(MPS_FRAMEWORKS) $< -o $@

$(BLAS_LIB): $(BLAS_SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(ACC_FRAMEWORK) $< -o $@

$(BLAS_MUL_LIB): $(BLAS_MUL_SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(ACC_FRAMEWORK) $< -o $@

clean:
	rm -f $(MUL_LIB) $(MPS_LIB) $(BLAS_LIB) $(BLAS_MUL_LIB)
	rm -rf __pycache__ *.pyc
