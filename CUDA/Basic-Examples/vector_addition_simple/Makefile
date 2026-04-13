#Makefile

# CUDA code generation flags

OBJ = vector_addition_1B
SRC = $(OBJ).cu

$(OBJ):$(SRC)
	nvcc   -O3 -m64 $(SRC) -o $(OBJ)

clean: 
	rm $(OBJ) *.o *~

