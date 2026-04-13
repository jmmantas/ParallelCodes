/*
This CUDA C program is an vector sum reduction example which includes 
three different approaches for the Block reduction. 

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
#include <string.h>
#include <time.h>



//*************************************************
// Function for checking CUDA runtime API results
//*************************************************
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

//*************************************************
// GPU WARMUP KERNEL     
// ************************************************
__global__ void warm_up_gpu(){
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float i,j=1.0,k=2.0;
  i = j+k; 
  j+=i+float(tid);
}

const int   N=1000000; // Size of the vector to be reduced

#define Bsize_sum   64


using namespace std;



//**************************************************************************
// Vector addition kernel 1: 1 element per thread 
//**************************************************************************
__global__ void reduce_Sum_1 (float *d_Vin, float *d_Vout, int N){

  extern __shared__ float sdata[];         // Dynamically allocated Shared memory array to store data                    
  int tid = threadIdx.x;                                 // Local index to access shared memory array sdata
  int i = blockIdx.x*blockDim.x + threadIdx.x; //Global index i to access the vector d_V

  // Load data into shared memory
  sdata[tid] = ((i < N) ? d_Vin[i] : 0.0f);
  __syncthreads();   // Barrier to ensure that all the warps in the block have completed the load

  // Shared memory reduction
  for (int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) 
        {sdata[tid] += sdata[tid + s];}
    __syncthreads();
  }
  // Write result for this block to global memory
  if (tid == 0)  d_Vout[blockIdx.x] = sdata[0]; 
}


//**************************************************************************
// Vector addition kernel 2: 2 elements per thread
//**************************************************************************
__global__ void reduce_Sum_2 (float *d_Vin, float *d_Vout, int N){

  extern __shared__ float sdata[];         // Dynamically allocated Shared memory array to store data                    
  int tid = threadIdx.x;                           // Local index to access shared memory array sdata
  int first = blockIdx.x*blockDim.x + threadIdx.x; //Global index to access 1st element in d_Vin
  int second=first +blockDim.x*gridDim.x;          //Global index to access 2nd element in d_Vin

  // Load data into shared memory
  sdata[tid] = d_Vin[first];
  sdata[tid] = (second < N) ? sdata[tid] + d_Vin[second] : sdata[tid];
  __syncthreads();   // Barrier to ensure that all the warps in the block have completed the load

  // Shared memory reduction
  for (int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) 
        {sdata[tid] += sdata[tid + s];}
    __syncthreads();
  }
  // Write result for this block to global memory
  if (tid == 0)  d_Vout[blockIdx.x] = sdata[0]; 
}



//**************************************************************************
// Vector addition kernel 3: 2 elements per thread with final loop unrolling 
//**************************************************************************

__device__ void warpReduce(volatile float* sdata, int tid) { 
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
  }

__global__ void reduce_Sum_3 (float *d_Vin, float *d_Vout, int N){

  extern __shared__ float sdata[];         // Dynamically allocated Shared memory array to store data                    
  int tid = threadIdx.x;                           // Local index to access shared memory array sdata
  int first = blockIdx.x*blockDim.x + threadIdx.x; //Global index to access 1st element in d_Vin
  int second=first +blockDim.x*gridDim.x;          //Global index to access 2nd element in d_Vin

  // Load data into shared memory
  sdata[tid] = d_Vin[first];
  sdata[tid] = (second < N) ? sdata[tid] + d_Vin[second] : sdata[tid];
  __syncthreads();   // Barrier to ensure that all the warps in the block have completed the load

  // Shared memory reduction
  for (int s=blockDim.x/2; s>32; s>>=1) {
    if (tid < s)  sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  // Final unrolling without synchronization (all threads are in the same warp)
  if (tid < 32){ 
    warpReduce(sdata, tid);
  }
    
  // Write result for this block to global memory
  if (tid == 0)  d_Vout[blockIdx.x] = sdata[0]; 
}




//**************************************************************************
int main()
//**************************************************************************

{


cout<<"********************************************* Warming up GPU!!!"<<endl;
// Warm up GPU
warm_up_gpu<<<(10000+Bsize_sum-1)/Bsize_sum, Bsize_sum>>>();


srand(time(NULL));
/* pointers to host memory */
float *a;
/* pointers to device memory */
float *a_d;

/* Allocate arrays a on host*/
a = (float*) malloc(N*sizeof(float));

/* Allocate arrays a_d on device*/
checkCuda(cudaMalloc ((void **) &a_d, sizeof(float)*N));

/* Initialize input array a*/
for (int i=0; i<N;i++){
  a[i]= (float) i;//(rand()%1000);
}

/* Copy data from host memory to device memory */
checkCuda(cudaMemcpy(a_d, a, sizeof(float)*N, cudaMemcpyHostToDevice));

cout <<endl<<"REDUCTION USING GPU OF A VECTOR WITH "<<N<<" ELEMENTS"<<endl;

//**************************************************
// Reduction 1 on GPU
//**************************************************
dim3 threadsPerBlock(Bsize_sum, 1);
dim3 numBlocks( ceil ((float)(N)/threadsPerBlock.x), 1);

// Sum vector on CPU
float * vsum1;
vsum1 = (float*) malloc(numBlocks.x*sizeof(float));

// Sum array  to be computed on GPU
float *vsum1_d; 
checkCuda(cudaMalloc ((void **) &vsum1_d, sizeof(float)*numBlocks.x));

int smemSize1 = threadsPerBlock.x*sizeof(float);


// Take initial time
double  t1=clock();

// Kernel launch to compute Sum array
reduce_Sum_1<<<numBlocks, threadsPerBlock, smemSize1>>>(a_d,vsum1_d, N);
cudaDeviceSynchronize();
double Tgpu1=clock();
Tgpu1=(Tgpu1-t1)/CLOCKS_PER_SEC;
checkCuda(cudaGetLastError());


/* Copy data from device memory to host memory */
checkCuda(cudaMemcpy(vsum1, vsum1_d, numBlocks.x*sizeof(float),cudaMemcpyDeviceToHost));

// *******   Perform final reduction in CPU *************
float sum_gpu1 = 0.0f;
for (int i=0; i<numBlocks.x; i++) {
  sum_gpu1 =sum_gpu1+vsum1[i]; 
}
cout<<endl<<"... Sum on GPU (Reduction kernel 1) ="<<sum_gpu1<<"               ";



//**************************************************
// Reduction 2 on GPU
//**************************************************
dim3 threadsPerBlock2(Bsize_sum, 1);
int N2 = (N+1)/2; 
dim3 numBlocks2( (N2+threadsPerBlock2.x-1)/threadsPerBlock2.x, 1);

// Sum vector on CPU
float * vsum2;
vsum2 = (float*) malloc(numBlocks2.x*sizeof(float));

// Sum vector  to be computed on GPU
float *vsum2_d; 
checkCuda(cudaMalloc ((void **) &vsum2_d, sizeof(float)*numBlocks2.x));

int smemSize2 = threadsPerBlock2.x*sizeof(float);


// Take initial time
double  t2=clock();

// Kernel launch to compute Sum array
reduce_Sum_2 <<<numBlocks2, threadsPerBlock2, smemSize2>>>(a_d,vsum2_d, N);

cudaDeviceSynchronize();
double Tgpu2=clock();
Tgpu2=(Tgpu2  -t2)/CLOCKS_PER_SEC;
checkCuda(cudaGetLastError());


/* Copy data from device memory to host memory */
checkCuda(cudaMemcpy(vsum2, vsum2_d, numBlocks2.x*sizeof(float),cudaMemcpyDeviceToHost));

// *******   Perform final reduction in CPU *************
float sum_gpu2 = 0.0f;
for (int i=0; i<numBlocks2.x; i++) {
  sum_gpu2 =sum_gpu2+vsum2[i]; 
}
cout<<endl<<"... Sum on GPU (Reduction kernel 2) ="<<sum_gpu2 <<"               ";






//**************************************************
// Reduction 3 on GPU
//**************************************************

// Take initial time
double  t3=clock();

// Kernel launch to compute Sum array
reduce_Sum_3 <<<numBlocks2, threadsPerBlock2, smemSize2>>>(a_d,vsum2_d, N);

cudaDeviceSynchronize();
double Tgpu3=clock();
Tgpu3=(Tgpu3  -t3)/CLOCKS_PER_SEC;
checkCuda(cudaGetLastError());

/* Copy data from device memory to host memory */
checkCuda(cudaMemcpy(vsum2, vsum2_d, numBlocks2.x*sizeof(float),cudaMemcpyDeviceToHost));

// *******   Perform final reduction in CPU *************
float sum_gpu3 = 0.0f;
for (int i=0; i<numBlocks2.x; i++){
  sum_gpu3 =sum_gpu3+vsum2[i]; 
  //cout<<"vsum2["<<i<<"]="<<vsum2[i]<<"    ";
}
cout<<endl<<"... Sum on GPU (Reduction kernel 3) ="<<sum_gpu3 <<"               ";

cout<<endl<<endl;









//***********************
// Compute sum on CPU
//***********************
cout <<endl<<"REDUCTION ON CPU OF A VECTOR WITH "<<N<<" ELEMENTS"<<endl;
// Take initial time
double  t_cpu0=clock();
float sum_cpu=0.0f;
for (int i=0; i<N;i++){
  sum_cpu = sum_cpu + a[i]; 
}
double Tcpu=clock();
Tcpu=(Tcpu-t_cpu0)/CLOCKS_PER_SEC;

cout<<endl<<"... Sum on CPU="<< sum_cpu<<endl;



cout<<endl<<endl<<"*********** Comparing results: *********** "<<endl;


cout<< "CPU Reduction Time=          "<<Tcpu<< "  seconds."<<endl;

cout<< "GPU Reduction Kernel 1 Time= "<<Tgpu1<< "  seconds."<<endl;

cout<< "GPU Reduction Kernel 2 Time= "<<Tgpu2<< "  seconds."<<endl;

cout<< "GPU Reduction Kernel 3 Time= "<<Tgpu3<< "  seconds."<<endl;

/* Free the memory */
free(a); free(vsum1); free(vsum2);
checkCuda(cudaFree(a_d)); 
checkCuda(cudaFree(vsum1_d));
checkCuda(cudaFree(vsum2_d));
}
