
/*
This file includes the header of the Graph class, which implements a graph using an adjacency matrix. 
The class provides methods to set the number of vertices, insert edges, retrieve edge values, 
print the adjacency matrix, and read the graph from a file.

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

//**************************************************************************
#ifndef GRAPH_H
#define GRAPH_H

//**************************************************************************
const int INF= 1000000;
//**************************************************************************
class Graph //Adjacency List class
{
	private:
		int *A;
	public:
        Graph(); // constructor
		~Graph(); // destructor
	    int vertices;
		//**************************************************************************
		// Set the number of vertices in the graph and initialize the adjacency matrix
	    void set_nverts(const int verts);

		//**************************************************************************
	    // Insert edge A->B in the Graph
		void insert_edge(const int ptA,const int ptB, const int edge);
		//**************************************************************************
		// Return the value of the edge A->B in the Graph
	    int edge(const int ptA,const int ptB);

		//**************************************************************************
		// Print the values stored in the adjacency matrix
        void print();

		//**************************************************************************
		// Read the graph from a file and populate the adjacency matrix
        void read(char *filename);

		//**************************************************************************
		// Return the adjacency matrix as a pointer to an integer array
		int * Get_Matrix(){return A;}
		//**************************************************************************
};

//**************************************************************************
#endif
//**************************************************************************
