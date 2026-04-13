/*
This CUDA C program is an very simple vector addition example 
using only one CUDA Block of Threads. 

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




__global__ void VecAdd( float *A, float *B, float *C, int N)
{
int i=threadIdx.x;
if (i<N)
  {
    C[i]=A[i]+B[i];
    printf("Thread %i::  C[%i] = A[%i]+B[%i] = %f \n",i,i,i,i,C[i] ); 
  }
}




int main(int argc, char* argv[])
{ 
   const int N=32; // Size of the vectors   

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
   for (int i=0; i<N;i++)
   {
     a[i]= (float) 2*i;
     b[i]= -(float) i;
   }

   /* Copy data from host memory to device memory */
   cudaMemcpy(a_d, a, sizeof(float)*N, cudaMemcpyHostToDevice);
   cudaMemcpy(b_d, b, sizeof(float)*N, cudaMemcpyHostToDevice);
   
   
   cout<<"************  LAUNCHING A NAIVE KERNEL TO ADD VECTORS USING ONE CUDA BLOCK"<<endl<<endl;
   /* Add arrays a and b, store result in c */
   VecAdd<<< 1, N >>>(a_d, b_d, c_d, N);
   /* Copy data from device memory to host memory */
   cudaMemcpy(c, c_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
   cout<<endl<<"END OF EXECUTION:   SHOWING  RESULT VECTOR C:";

   
   /* Print c */
   cout<<endl<<"C = { ";
   for (int i=0; i<N;i++)
     cout<<"  "<<c[i];
   cout<<" }"<<endl;   

   /* Free the host and device  memory */
   free(a); free(b); free(c);
   cudaFree(a_d); cudaFree(b_d);cudaFree(c_d);

}
