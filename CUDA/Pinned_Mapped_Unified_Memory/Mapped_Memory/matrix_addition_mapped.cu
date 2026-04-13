/*
This CUDA C program is an matrix addition example using mapped memory. 

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
#include <time.h>

using namespace std;

const int N=10;

//************************************************************* 
// Matrix Addition CUDA Kernel called by MatAdd()
// Computes C = A + B
// where A, B and C are NxN matrices stored in row-major order
//*************************************************************
__global__ void MatAdd( float *A, float *B, float *C, int N){
//**********************************************
  int j = blockIdx.x * blockDim.x + threadIdx.x; // row index
  int i = blockIdx.y * blockDim.y + threadIdx.y; // column index
  int index=i*N+j; // Compute global 1D index

  if (i < N && j < N)   {
	  C[index] = A[index] + B[index]; // Compute C element
  }
}
//**************************************************************


//**************************************************************
int main(int argc, char* argv[]){ 
//**************************************************************

  const int NN=N*N;
  const int size_in_bytes=NN*sizeof(float);

  // Variable to store kernel execution time
  float time_kernel;

   // Events for timing
  cudaEvent_t start, stop; 
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

/* pointers to pinned host memory */
  float * A_p, *B_p, *C;

/* pointers to device memory  */
  float * A_d, *B_d, *C_d;
 

  /* Allocate arrays A_p and B_p on pinned host memory*/
  cudaHostAlloc( (void**)&A_p, size_in_bytes,cudaHostAllocMapped);
  cudaHostAlloc( (void**)&B_p, size_in_bytes,cudaHostAllocMapped);

  /* Allocate output array C on host memory*/
  C=(float *) malloc(size_in_bytes);

  /* Allocate output array C_d on device memory*/
  cudaMalloc( (void**)&C_d, size_in_bytes);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
  /* Initialize mapped arrays */
   for (int i=0; i<N;i++)
    for (int j=0;j<N;j++)
      {
       A_p[i*N+j]=(float) -i;
       B_p[i*N+j]=(float) (2*i);
      };

  cudaHostGetDevicePointer( (void **) &A_d,  (void *) A_p, 0);
  cudaHostGetDevicePointer( (void **) &B_d,  (void *) B_p, 0);


  /* Compute the execution configuration */
  dim3 threadsPerBlock (16, 16);
  dim3 numBlocks( ceil ((float)(N)/threadsPerBlock.x), ceil ((float)(N)/threadsPerBlock.y) );
  
  // Kernel Launch
  cudaEventRecord(start, 0);
  MatAdd <<<numBlocks, threadsPerBlock>>> (A_d, B_d, C_d, N);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_kernel, start, stop);

 
  cout <<"...................................................."<<endl;
  cout <<"Kernel Execution Time="<<time_kernel<<" ms"<<endl;
  cout <<"...................................................."<<endl;
  
  
/* Copy data from device memory to host memory */
cudaMemcpy(C, C_d, sizeof(float)*NN, cudaMemcpyDeviceToHost);

// Print the output matrix C
cout<< "C=   ";
for (int i=0; i<N;i++)
    for (int j=0;j<N;j++)
      {
       cout <<C[i*N+j]<<" ,";
      }
cout<<endl;
  
 
/* Free host and device memory */
cudaFreeHost(A_p);  cudaFreeHost(B_p); cudaFree(C_d);

 
return 0;
}