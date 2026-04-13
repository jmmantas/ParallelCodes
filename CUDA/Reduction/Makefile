#Makefile

# CUDA code generation flags

GENCODE_SM80    := #-gencode arch=compute_80,code=compute_80
GENCODE_FLAGS   ?= $(GENCODE_SM80) 

OBJ = vector_reduction
SRC = $(OBJ).cu

$(OBJ):$(SRC)
	nvcc   -O3 -m64   $(GENCODE_FLAGS) $(SRC) -o $(OBJ)

clean: 
	rm $(OBJ) *.o *~

