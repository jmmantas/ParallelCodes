/*
This CUDA C program is an simple matrix addition example 
using 2D CUDA Blocks. 

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

__global__ void MatAdd( float *A, float *B, float *C, int N)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;  // Compute row index
  int i = blockIdx.y * blockDim.y + threadIdx.y;  // Compute column index
  int index=i*N+j; // Compute global 1D index

  if (i < N && j < N)
     {
	    printf( "Block (%d,%d), Thread (%d,%d), i=%d, j=%d \n",  blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,i,j);
	    C[index] = A[index] + B[index]; // Compute C element
     }
}

int main(int argc, char* argv[]){ 
  const int NN=N*N;
  /* pointers to host memory */
  /* Allocate arrays A, B and C on host*/
  float * A = (float*) malloc(NN*sizeof(float));
  float * B = (float*) malloc(NN*sizeof(float));
  float * C = (float*) malloc(NN*sizeof(float));

  /* pointers to device memory */
  float *A_d, *B_d, *C_d;
  /* Allocate arrays a_d, b_d and c_d on device*/
  cudaMalloc ((void **) &A_d, sizeof(float)*NN);
  cudaMalloc ((void **) &B_d, sizeof(float)*NN);
  cudaMalloc ((void **) &C_d, sizeof(float)*NN);

  /* Initialize arrays a and b */
   for (int i=0; i<N;i++)
    for (int j=0;j<N;j++)
      {
       A[i*N+j]=(float) i; 
       B[i*N+j]=(float) (1.0f-i);
      };

  /* Copy data from host memory to device memory */
  cudaMemcpy(A_d, A, sizeof(float)*NN, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, sizeof(float)*NN, cudaMemcpyHostToDevice);

  /* Compute the execution configuration */
  dim3 threadsPerBlock (8, 8);
  dim3 numBlocks( ceil ((float)(N)/threadsPerBlock.x), ceil ((float)(N)/threadsPerBlock.y) );
  
  double  time1=clock();
  // Kernel Launch
  MatAdd <<<numBlocks, threadsPerBlock>>> (A_d, B_d, C_d, N);
  cudaDeviceSynchronize();
  double time2=clock();
  double time=(time2-time1)/CLOCKS_PER_SEC;

  /* Copy data from deveice memory to host memory */
  cudaMemcpy(C, C_d, sizeof(float)*NN, cudaMemcpyDeviceToHost);

  /* Print c */
  for (int i=0; i<N;i++){
    for (int j=0;j<N;j++){
      cout <<"C["<<i<<","<<j<<"]="<<C[i*N+j]<< ",  ";
    }
    cout<<endl;
  }
      
 
  cout <<"Kernel Execution Time="<<time<<endl;
  
  /* Free the memory */
  free(A); free(B); free(C);
  cudaFree(A_d); cudaFree(B_d);cudaFree(C_d);

}
