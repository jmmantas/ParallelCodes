/*
This CUDA C program is an vector addition example. 

ParallelCodes Copyright (C) 2026 Jose Miguel Mantas Ruiz (jmmantas@ugr.es)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#include <iostream>
#include <fstream>


using namespace std;

// Declaring the size of the vectors and the block size for the kernel launch
const int N=70, BSIZE=64;

/**************************************************************/
// Warm up Kernel
/**************************************************************/
__global__ void warm_up_gpu(){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    // Simple arithmetic to keep cores busy
    ia = ib = 0.0f;
    ib += ia + tid;
}
/**************************************************************/

/**************************************************************/
// Standard Vector Addition Kernel with coalesced and aligned access
/**************************************************************/
__global__ void VecAdd(float* A, float* B, float* C, int N){ 
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i<N) {    
	  C[i] = A[i] + B[i];
    printf("Block %i,   Thread %i::  C[%i] = A[%i]+B[%i] = %f\n"
                            ,blockIdx.x,threadIdx.x,i,i,i,C[i] );     
  }
}
/**************************************************************/

/**************************************************************/
//Vector Addition Kernel with irregular access to A and B arrays
/**************************************************************/
__global__ void VecAdd_Irr (float* A, float* B, float* C, int N) { 
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int ip32=i+32; int i0=blockIdx.x * blockDim.x;
   if (i<N)      
       if (threadIdx.x==0 && ip32<N) C[ip32] = A[ip32] + B[ip32];
       else if (threadIdx.x==32) C[i0] = A[i0] + B[i0];
       else C[i] = A[i] + B[i];
}
/**************************************************************/

int main(int argc, char* argv[])
{
  /* pointers to host memory */
  float *A, *B, *C;
  /* pointers to device memory */
  float *A_d, *B_d, *C_d;

  /* Allocate arrays a, b and c on host*/
  A = (float*) malloc(N*sizeof(float));
  B = (float*) malloc(N*sizeof(float));
  C = (float*) malloc(N*sizeof(float));

  /* Allocate arrays a_d, b_d and c_d on device*/
  cudaMalloc ((void **) &A_d, sizeof(float)*N);
  cudaMalloc ((void **) &B_d, sizeof(float)*N);
  cudaMalloc ((void **) &C_d, sizeof(float)*N);

  /* Initialize arrays a and b */
  for (int i=0; i<N;i++){
    A[i]= (float) 2*i;
    B[i]= -(float) i;
  }
 
  cout<< "...Warming up GPU"<<endl;
  /* Copy data from host memory to device memory */
  warm_up_gpu<<< (N+BSIZE-1)/BSIZE, BSIZE >>>();

  cout<< "... Vectors A and B  are copied from the host mem. to the Device mem. (A_d and B_d)"<<endl;
  /* Copy data from host memory to device memory */
  cudaMemcpy(A_d, A, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, sizeof(float)*N, cudaMemcpyHostToDevice);

  /* Compute the execution configuration */
  cout<< "... Kernel is launched to add vectors with Block size= " <<BSIZE<<endl;
  /* Add arrays a and b, store result in c */
  VecAdd<<< (N+BSIZE-1)/BSIZE, BSIZE >>>(A_d, B_d, C_d, N);

  cout<< "... Vector C_d   is copied from the host mem. to the Device mem. (C)"<<endl;
  /* Copy data from deveice memory to host memory */
  cudaMemcpy(C, C_d, sizeof(float)*N, cudaMemcpyDeviceToHost);

  /* Print C */
  cout<<endl<<endl<<"The result is  C = { ";
  for (int i=0; i<N;i++) cout<<"  "<<C[i];
  cout<<" }"<<endl;   

  // Free the memory
  free(A); free(B); free(C);
  cudaFree(A_d); cudaFree(B_d);cudaFree(C_d);

}
