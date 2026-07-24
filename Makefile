CC = /usr/local/cuda/bin/nvcc
TORCH_HOME = $(shell python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')
CFLAGS = -I$(TORCH_HOME)/include \
				 -I$(TORCH_HOME)/include/torch/csrc/api/include \
				 -lcublas

HELLO = build/hello
HELLO_SRC = csrc/hello_world.cu

BASIC = build/basic
BASIC_SRC = csrc/basic.cu

GEMM = build/gemm
GEMM_SRC = csrc/gemm.cu

all: $(HELLO) $(BASIC) $(GEMM)

$(HELLO): $(HELLO_SRC)
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

$(BASIC): $(BASIC_SRC)
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@ 

$(GEMM): $(GEMM_SRC)
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -rf build

.PHONY: all clean
