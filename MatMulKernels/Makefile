NVCC=nvcc
CFLAGS=-O3
SRC=main.cu sgemm1.cu sgemm_coalescing.cu sgemm_smem.cu
OUT=sgemm

all: $(OUT)

$(OUT): $(SRC)
	$(NVCC) $(CFLAGS) $(SRC) -o $(OUT)

run: $(OUT)
	./$(OUT)

clean:
	rm -f $(OUT)
