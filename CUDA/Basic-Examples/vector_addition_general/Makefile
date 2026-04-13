#Makefile

# CUDA code generation flags

GENCODE_FLAGS   = 
OBJ = vector_addition
SRC = $(OBJ).cu

$(OBJ):$(SRC)
	nvcc -O3 -m64   $(GENCODE_FLAGS) $(SRC) -o $(OBJ)

clean: 
	rm $(OBJ) *.o *~

