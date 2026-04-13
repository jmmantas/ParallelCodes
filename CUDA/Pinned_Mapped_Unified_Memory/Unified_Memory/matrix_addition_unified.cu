/*
This CUDA C program is an matrix addition example which uses unified memory. 

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

  /* pointers to managed memory */
  float *A, *B, *C;

  /* Allocate managed arrays A, B and C*/
  cudaMallocManaged( (void**)&A, size_in_bytes,cudaMemAttachGlobal);
  cudaMallocManaged( (void**)&B, size_in_bytes,cudaMemAttachGlobal);
  cudaMallocManaged( (void**)&C, size_in_bytes,cudaMemAttachGlobal);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
  /* Initialize arrays A and B */
   for (int i=0; i<N;i++)
    for (int j=0;j<N;j++)
      {
       A[i*N+j]=(float) -i;
       B[i*N+j]=(float) (2*i);
      };


  /* Compute the execution configuration */
  dim3 threadsPerBlock (16, 16);
  dim3 numBlocks( ceil ((float)(N)/threadsPerBlock.x), ceil ((float)(N)/threadsPerBlock.y) );
  
  // Kernel Launch
  cudaEventRecord(start, 0);
  MatAdd <<<numBlocks, threadsPerBlock>>> (A, B, C, N);
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_kernel, start, stop);

  cout <<"...................................................."<<endl;
  cout <<"Kernel Execution Time="<<time_kernel<<" ms"<<endl;
  cout <<"...................................................."<<endl;
  


  cout<< "C=   ";
  for (int i=0; i<N;i++){
    for (int j=0;j<N;j++)  {
      cout <<C[i*N+j]<<" ,";
    }  
  }
  cout<<endl;
   
  /* Free managed memory */
  cudaFree(A);  cudaFree(B); cudaFree(C);

  return 0;
}