/*
This code implements a Graph class using an adjacency matrix representation. 
It includes methods for setting the number of vertices, inserting edges, 
retrieving edge values, printing the adjacency matrix, and reading the graph from a file. 
The graph is represented as a one-dimensional array (A) where the value at index 
[i*vertices + j] corresponds to the edge from vertex i to vertex j.

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


//***********************************************************************
#include "Graph.h"
#include <iostream>
#include <fstream>

#include <stdlib.h>
#include <string.h>

using namespace std;

//***********************************************************************
Graph::Graph ()		// Constructor
{
}

//***********************************************************************
Graph::~Graph ()		// Destructor
{
  delete[] A;
}

//***********************************************************************
// Set the number of vertices in the graph and initialize the adjacency matrix
void Graph::set_nverts (const int nverts)
//***********************************************************************
{
  A=new int[nverts*nverts];
  vertices=nverts;
}


//***********************************************************************
// Insert edge A->B in the Graph
void Graph::insert_edge(const int vertA, const int vertB, const int edge){
  A[vertA*vertices+vertB]=edge;
}
//***********************************************************************
int Graph::edge(const int ptA,const int ptB)
{
  return A[ptA*vertices+ptB];
}


//***********************************************************************
// Print the values stored in the adjacency matrix
 void Graph::print()
//*********************************************************************** 
{
  int i,j,vij;
  for(i=0;i<vertices;i++){
    cout << "A["<<i << ",*]= ";
    for(j=0;j<vertices;j++){
      if (A[i*vertices+j]==INF) 
        cout << "INF";
      else  
        cout << A[i*vertices+j];
      if (j<vertices-1) 
        cout << ",";
      else
        cout << endl;
   }
 }
}

//***********************************************************************
// Read the graph from a file and populate the adjacency matrix
void Graph::read(char *filename) {
//***********************************************************************

  #define BUF_SIZE 100
  std::ifstream infile(filename);
  if (!infile)
	{
	  cerr << "Invalid File name \"" << filename << "\" !!" << endl;
	  cerr << "Exiting........." << endl;
	  exit(-1);
	}
  //Get the number of vertices
  char buf[BUF_SIZE];
  infile.getline(buf,BUF_SIZE,'\n');
  vertices=atoi(buf);
  A=new int[vertices*vertices];
 
  int i,j;
  for(i=0;i<vertices;i++)
     for(j=0;j<vertices;j++)
	 if (i==j) A[i*vertices+j]=0;
         else A[i*vertices+j]=INF;
    
  while (infile.getline(buf,BUF_SIZE) && infile.good() && !infile.eof())
	{
	  char *vertname2 = strpbrk(buf, " \t");
	  *vertname2++ = '\0';
	  char *buf2 = strpbrk(vertname2, " \t");
	  *buf2++ = '\0';
	  int weight = atoi(buf2);
	  i=atoi(buf);
	  j=atoi(vertname2);
    A[i*vertices+j]=weight;
	 }
}
