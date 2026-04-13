/*
This CUDA C program is an vector addition example using CUBLAS. 

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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

using namespace std;

const int N=150;

int main(int argc, char* argv[])
{
  cublasStatus_t status;  
  cublasHandle_t handle;

  /* pointers to host memory */
  float *a, *b, *c;
  /* pointers to device memory */
  float *a_d, *b_d, *c_d;

  /* Allocate arrays a, b and c on host*/
  a = (float*) malloc(N*sizeof(float));
  b = (float*) malloc(N*sizeof(float));
  c = (float*) malloc(N*sizeof(float));

  /* Allocate arrays a_d, b_d and c_d on device*/
  cudaMalloc ((void **) &a_d, sizeof(float)*N);
  cudaMalloc ((void **) &b_d, sizeof(float)*N);
  cudaMalloc ((void **) &c_d, sizeof(float)*N);

  /* Initialize arrays a and b */
  for (int i=0; i<N;i++){
    a[i]= (float) i;
    b[i]= -(float) 2*i;
  }

  /* Initialize CUBLAS */
  cout<<"Executing Simple CUBLAS Test ...."<<endl;
  status = cublasCreate(&handle);

  if (status != CUBLAS_STATUS_SUCCESS) {
    cerr<<"Inicialization Error in the CUBLAS handle creation!!!" <<endl;
    return EXIT_FAILURE;
  }

  /* Copy vectors from host memory to device memory */
  status = cublasSetVector(N, sizeof(a[0]), a, 1, a_d, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
     cerr<<"device access error (writing vector a)!!!"<<endl;
    return EXIT_FAILURE;
  }

  status = cublasSetVector(N, sizeof(b[0]), b, 1, b_d, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
     cerr<<"device access error (writing vector b)!!!"<<endl;
    return EXIT_FAILURE;
  }

  /* Performs vector operations using cublas */
  float alpha = 1.0f;
  status = cublasScopy(handle, N, b_d, 1, c_d, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
     cerr<<"dcopy error!!!"<<endl;
    return EXIT_FAILURE;
  }
  status = cublasSaxpy(handle, N, &alpha, a_d, 1, c_d, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
     cerr<<"daxpy error!!!"<<endl;
    return EXIT_FAILURE;
  }

  // Move the result to the host 
  status = cublasGetVector(N, sizeof(c[0]), c_d, 1, c, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    cerr<<"Error reading result from device (reading c_d)"<<endl;
    return EXIT_FAILURE;
  }
  
  /* Print c */
  cout<<endl<<endl<<"C = { ";
  for (int i=0; i<N;i++)
     cout<<"  "<<c[i];
  cout<<" }"<<endl;   

  // Destry the CUBLAS handle
  status = cublasDestroy(handle);

  if (status != CUBLAS_STATUS_SUCCESS) {
    cerr<<"Error destroying CUBLAS handle!!!" <<endl;
    return EXIT_FAILURE;
  }

  // Free the memory
  free(a); free(b); free(c);
  cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);

}
