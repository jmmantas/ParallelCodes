#Makefile

# CUDA code generation flags

OBJ = advection1d
SRC=$(OBJ).cu
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS   ?= $(GENCODE_SM60) $(GENCODE_SM70) $(GENCODE_SM80)


${OBJ}:${OBJ}.cu
	nvcc  -O3 -m64   $(GENCODE_FLAGS) ${SRC} -o ${OBJ}

clean: 
	rm ${OBJ} *.o *~
      