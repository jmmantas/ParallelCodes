/*
This CUDA C program is an matrix addition example using pinned memory to store 
operands and the result in host memory. 

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

// Variables to store execution times
  float time_pageable, time_pinned, time_kernel; 

  /* pointers to pageable host memory */
  float * A, *B, *C;

/* pointers to pinned host memory */
  float * A_p, *B_p, *C_p;


  /* Allocate arrays A, B and C on pageable host memory*/ 
  A=(float*) malloc(size_in_bytes);
  B=(float*) malloc(size_in_bytes);
  C=(float*) malloc(size_in_bytes);

 
  /* Allocate arrays A_p, B_p and C_p on pinned host memory*/
  cudaMallocHost( (void**)&A_p, size_in_bytes);
  cudaMallocHost( (void**)&B_p, size_in_bytes);
  cudaMallocHost( (void**)&C_p, size_in_bytes);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

  /* pointers to device memory */
  float *A_d, *B_d, *C_d;

  /* Allocate arrays a_d, b_d and c_d on device*/
  cudaMalloc ((void **) &A_d, sizeof(float)*NN);
  cudaMalloc ((void **) &B_d, sizeof(float)*NN);
  cudaMalloc ((void **) &C_d, sizeof(float)*NN);

  /* Initialize host arrays */
   for (int i=0; i<N;i++) {
    for (int j=0;j<N;j++)   {
       A[i*N+j]=(float) i; 
       B[i*N+j]=(float) (1-i);
       
       A_p[i*N+j]=A[i*N+j];
       B_p[i*N+j]=B[i*N+j];
       
    }
  }

  // Events for timing
  cudaEvent_t start, stop; 
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  
  /* Copy data from host memory to device memory using pageable memory */
  cudaEventRecord(start, 0);
  cudaMemcpy(A_d, A, sizeof(float)*NN, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, sizeof(float)*NN, cudaMemcpyHostToDevice);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_pageable, start, stop);

  /* Copy data from host memory to device memory using*/
  cudaEventRecord(start, 0);
  cudaMemcpy(A_d, A_p, sizeof(float)*NN, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_p, sizeof(float)*NN, cudaMemcpyHostToDevice);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_pinned, start, stop);

  cout <<"Pageable Host-Device Transfer Time (A and B) ="<<time_pageable<<"  ms"<<endl;
  cout <<"Pinned   Host-Device Transfer Time (A and B) ="<<time_pinned<<"  ms"<<endl;

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
  
  cudaEventRecord(start, 0);
  cudaMemcpy(C, C_d, sizeof(float)*NN, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_pageable, start, stop);

  cudaEventRecord(start, 0);
  cudaMemcpy(C_p, C_d, sizeof(float)*NN, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_pinned, start, stop);
  
  
  

  cout <<"Pageable Device-Host Transfer Time (C)= "<<time_pageable<<"  ms"<<endl;
  cout <<"Pinned   Device-Host Transfer Time (C)= "<<time_pinned<<"  ms"<<endl;
  cout<<endl;


  for(int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      {
       if ((C[i*N+j] != 1.0f) || (C_p[i*N+j] != C[i*N+j]))
         cout<<"Error: C["<<i<<"]["<<j<<"]="<<C[i*N+j]<<"  C_p["<<i<<"]["<<j<<"]="<<C_p[i*N+j]<<endl;
      }

   cout <<"Test PASSED"<<endl;

  // Free the pageable host memory
  free(A); free(B); free(C);

  // Free the pinned host memory
  cudaFreeHost(A_p);  cudaFreeHost(B_p); cudaFreeHost(C_p);

  // Free the device memory
  cudaFree(A_d); cudaFree(B_d);cudaFree(C_d);

  // Destroy Events
  cudaEventDestroy(start);cudaEventDestroy(stop);

}