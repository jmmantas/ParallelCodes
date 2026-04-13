/*
This CUDA C program implements the Floyd-Warshall algorithm for 
finding the shortest paths between all pairs of vertices in a 
weighted dense graph.


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
#include "Graph.h"
#include <assert.h>

#define DEBUG 1

using namespace std;

#define BLOCKSIZE 256

//*************************************************
// Function for checking CUDA runtime API results
//*************************************************
inline cudaError_t checkCuda(cudaError_t result) {
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


//**************************************************************************
__global__ void floyd_kernel(int * M, const int nverts, const int k) {
    int ij = threadIdx.x + blockDim.x * blockIdx.x;
    int i= ij / nverts;
    int j= ij - i * nverts;
    if (i<nverts && j< nverts) {
    int Mij = M[ij];
    if (i != j && i != k && j != k) {
	int Mikj = M[i * nverts + k] + M[k * nverts + j];
    Mij = (Mij > Mikj) ? Mikj : Mij;
    M[ij] = Mij;}
  }
}
//**************************************************************************

//**************************************************************************
// ************  MAIN FUNCTION *********************************************
int main (int argc, char *argv[]) {

    double time, Tcpu, Tgpu;

    if (argc != 2) {
	    cerr << "Syntax: " << argv[0] << " <Graph File>" << endl;
		return(-1);
	  }	

    //Get GPU information
    int devID=0;
    cudaDeviceProp props;
   
    cout<<"Using Device "<<devID<<endl;
    cout<<"....................................................."<<endl<<endl;
    checkCuda(cudaGetDeviceProperties(&props, devID));
    cout<<"****************************************************************************************"<<endl;
    cout<<"Using Device "<< devID<<": "<<props.name<<"  with CUDA Compute Capability "<<props.major<<"."<<props.minor<<endl;
    cout<<"****************************************************************************************"<<endl<<endl;

	  checkCuda(cudaSetDevice(devID)); 


    cout<<"********************************************* Warming up GPU!!!"<<endl;
    // Warm up GPU
    warm_up_gpu<<<(10000+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>();


	  // Declaration of the Graph object
	  Graph G;
	
	  // Read the Graph
	  G.read(argv[1]);

	//cout << "The input Graph:"<<endl;
	//G.print();
	const int nverts = G.vertices;
	const int niters = nverts;
	const int nverts2 = nverts * nverts;

	int *c_Out_M = new int[nverts2];
	int size = nverts2*sizeof(int);
	int * d_In_M = NULL;

	checkCuda(cudaMalloc((void **) &d_In_M, size));

    // Get the integer 2D array for the dense graph
	int *A = G.Get_Matrix();

    //**************************************************************************
	// GPU phase
	//**************************************************************************
	
    time=clock();

	checkCuda(cudaMemcpy(d_In_M, A, size, cudaMemcpyHostToDevice));

    // Main Loop
	for(int k = 0; k < niters; k++) {
		//printf("CUDA kernel launch \n");
	 	int threadsPerBlock = BLOCKSIZE;
	 	int blocksPerGrid = (nverts2 + threadsPerBlock - 1) / threadsPerBlock;
        // Kernel Launch
	    floyd_kernel<<<blocksPerGrid,threadsPerBlock >>>(d_In_M, nverts, k);
	    checkCuda(cudaGetLastError());
	}
	checkCuda(cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost));

	Tgpu=(clock()-time)/CLOCKS_PER_SEC;
	
	cout << "Time spent on GPU= " << Tgpu << endl << endl;

    //**************************************************************************
	// CPU phase
	//**************************************************************************

	time=clock();

	// BUCLE PPAL DEL ALGORITMO
	int inj, in, kn;
	for(int k = 0; k < niters; k++) {
          kn = k * nverts;
	  for(int i=0;i<nverts;i++) {
			in = i * nverts;
			for(int j = 0; j < nverts; j++)
	       			if (i!=j && i!=k && j!=k){
			 	    inj = in + j;
			 	    A[inj] = min(A[in+k] + A[kn+j], A[inj]);
	       }
	   }
	}
  
  Tcpu=(clock()-time)/CLOCKS_PER_SEC;
  cout << "Time spent on CPU= " << Tcpu << endl << endl;
  cout<<"....................................................."<<endl<<endl;

  cout << "Speedup TCPU/TGPU= " << Tcpu / Tgpu << endl;
  cout<<"....................................................."<<endl<<endl;

  
  bool errors=false;
  // Error Checking (CPU vs. GPU)
  for(int i = 0; i < nverts; i++)
    for(int j = 0; j < nverts; j++)
       if (abs(c_Out_M[i*nverts+j] - G.edge(i,j)) > 0)
         {cout << "Error (" << i << "," << j << ")   " << c_Out_M[i*nverts+j] << "..." << G.edge(i,j) << endl;
		  errors=true;
		 }


  if (!errors){ 
    cout<<"....................................................."<<endl;
	cout<< "WELL DONE!!! No errors found ............................"<<endl;
	cout<<"....................................................."<<endl<<endl;

  }
  checkCuda(cudaFree(d_In_M));
  delete[] c_Out_M;

}

