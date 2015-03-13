#include "mpi.h"
#include "grid.h"
#include "tensor.h"
#include "contraction.h"
#include "redistribute.h"
#include "grid_redib.h"
using namespace RRR;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank,np;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    MPI_Comm_size(MPI_COMM_WORLD,&np);
	
	 
	 int ngrids=2;
	 
	 int* grid_sizes = new int[ngrids];
	 
	 
	 grid_sizes[0]=16;
	 grid_sizes[1]=16;
	 
	 if(rank==0) cout<<"Grid about to initialized"<<endl;
 int size = 16;
    int block_grid = 8;
    int grid_dim = 4;
    int* pgrid = new int[grid_dim];

    
	pgrid[0] = 2;
	pgrid[1] = 2;
	pgrid[2] = 2;
	pgrid[3] = 2;
    
int* pgrid2 = new int[grid_dim];

    
	pgrid2[0] = 2;
	pgrid2[1] = 2;
	pgrid2[2] = 2;
	pgrid2[3] = 2;
    
	 
	/* for(int i=0; i<ngrids; i++)
	 {	
		int start_rank=0;
		 int* proc_ranks = new int[grid_sizes[i]];
		 for(int j=0;j<i;j++) start_rank+=grid_sizes[j];
		 for(int j=0;j<grid_sizes[i];j++) proc_ranks[j]=start_rank+j;
		 MPI_Group orig_group, new_group; 
		 MPI_Comm new_comm; 

		 MPI_Comm_group(MPI_COMM_WORLD, &orig_group); 
		 MPI_Group_incl(orig_group, grid_sizes[i], proc_ranks, &new_group);
		 MPI_Comm_create(MPI_COMM_WORLD, new_group, &grid_comms[i]);
	 }*/
	 MPI_Group orig_group, new_group; 
		 MPI_Comm new_comm[2]; 
		 
		 if(rank>=0 && rank<grid_sizes[0]){
	 int* proc_ranks = new int[grid_sizes[0]];
		 
		 for(int j=0;j<grid_sizes[0];j++) proc_ranks[j]=j;
		 

		 MPI_Comm_group(MPI_COMM_WORLD, &orig_group); 
		
		 MPI_Group_incl(orig_group, grid_sizes[0], proc_ranks, &new_group);
		
		 }
		 
		 else
		 { 

			MPI_Comm_group(MPI_COMM_WORLD, &orig_group);
			int* proc_ranks = new int[grid_sizes[0]];
			proc_ranks[0]=rank;
			 MPI_Group_incl(orig_group, 1, proc_ranks, &new_group);
			 
		 }
		 MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm[0]);
		 
		 
		 if(rank>=grid_sizes[0]){
	 int* proc_ranks = new int[grid_sizes[1]];
		 
		 for(int j=0;j<grid_sizes[1];j++) proc_ranks[j]=16+j;
		 

		 MPI_Comm_group(MPI_COMM_WORLD, &orig_group); 
		
		 MPI_Group_incl(orig_group, grid_sizes[1], proc_ranks, &new_group);
		
		 }
		 
		 else
		 { 

			MPI_Comm_group(MPI_COMM_WORLD, &orig_group);
			int* proc_ranks = new int[grid_sizes[0]];
			proc_ranks[0]=rank;
			 MPI_Group_incl(orig_group, 1, proc_ranks, &new_group);
			 
		 }
		 
		 MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm[1]);
		 
		 
	
	
	
	//if(rank<grid_sizes[0])
		
	//{
	 Grid* grid = new Grid(grid_dim, pgrid,new_comm[0],0,ngrids,grid_sizes);
   
    
	
	 int* size_A = new int[4];
    int* block_grid_A = new int[4];
    int* idmap_A = new int[4];

    size_A[0] = size;
    size_A[1] = size;
    size_A[2] = size;
    size_A[3] = size;

    block_grid_A[0] = block_grid;
    block_grid_A[1] = block_grid;
    block_grid_A[2] = block_grid;
    block_grid_A[3] = block_grid;

    idmap_A[0] = 0;
    idmap_A[1] = 1;
    idmap_A[2] = 2;
    idmap_A[3] = 3;

    Tensor* A = new Tensor("cccc", idmap_A, size_A, block_grid_A, grid);
    A->initialize();

    if(rank == 0) cout<<"Tensor A initialized"<<endl;

    int* size_B = new int[4];
    int* block_grid_B = new int[4];
    int* idmap_B = new int[4];

    size_B[0] = size;
    size_B[1] = size;
    size_B[2] = size;
    size_B[3] = size;

    block_grid_B[0] = block_grid;
    block_grid_B[1] = block_grid;
    block_grid_B[2] = block_grid;
    block_grid_B[3] = block_grid;

    idmap_B[0] = 2;
    idmap_B[1] = 3;
    idmap_B[2] = 0;
    idmap_B[3] = 1;

    Tensor* B = new Tensor("cccc", idmap_B, size_B, block_grid_B, grid);
    B->initialize();

    if(rank == 0) cout<<"Tensor B initialized"<<endl;

    int* size_C = new int[4];
    int* block_grid_C = new int[4];
    int* idmap_C = new int[4];

    size_C[0] = size;
    size_C[1] = size;
    size_C[2] = size;
    size_C[3] = size;

    block_grid_C[0] = block_grid;
    block_grid_C[1] = block_grid;
    block_grid_C[2] = block_grid;
    block_grid_C[3] = block_grid;

    idmap_C[0] = 0;
    idmap_C[1] = 1;
    idmap_C[2] = 2;
    idmap_C[3] = 3;

    Tensor* C = new Tensor("cccc", idmap_C, size_C, block_grid_C, grid);
    C->initialize();

    if(rank == 0) cout<<"Tensor C initialized"<<endl;

    double time =0.0;
    if(rank==0) cout<<"C [a,b,m,n ] = A [a,b,k,l ] x B [m,n,k,l ]"<<endl;
    Contraction* C0 = new Contraction(A, B, C, grid);
    time =-MPI_Wtime();
    C0->contract( "a,b,k,l", "m,n,k,l", "a,b,m,n");
	//}
	
	//if(rank>=grid_sizes[0])
		
	//{
	 Grid* grid2 = new Grid(grid_dim, pgrid2,new_comm[1],1,ngrids,grid_sizes);
   GridRedistribute* Credib = new GridRedistribute(C,idmap_B,grid2);
	Credib->redistribute();	
    
	/*
	 int* size_A2 = new int[4];
    int* block_grid_A2 = new int[4];
    int* idmap_A2 = new int[4];

    size_A2[0] = size;
    size_A2[1] = size;
    size_A2[2] = size;
    size_A2[3] = size;

    block_grid_A2[0] = block_grid;
    block_grid_A2[1] = block_grid;
    block_grid_A2[2] = block_grid;
    block_grid_A2[3] = block_grid;

    idmap_A2[0] = 0;
    idmap_A2[1] = 1;
    idmap_A2[2] = 2;
    idmap_A2[3] = 3;

    Tensor* A2 = new Tensor("cccc", idmap_A2, size_A2, block_grid_A2, grid2);
    A2->initialize();

    if(rank == 16) cout<<"Tensor A2 initialized"<<endl;

    int* size_B2 = new int[4];
    int* block_grid_B2 = new int[4];
    int* idmap_B2 = new int[4];

    size_B2[0] = size;
    size_B2[1] = size;
    size_B2[2] = size;
    size_B2[3] = size;

    block_grid_B2[0] = block_grid;
    block_grid_B2[1] = block_grid;
    block_grid_B2[2] = block_grid;
    block_grid_B2[3] = block_grid;

    idmap_B2[0] = 2;
    idmap_B2[1] = 3;
    idmap_B2[2] = 0;
    idmap_B2[3] = 1;

    Tensor* B2 = new Tensor("cccc", idmap_B2, size_B2, block_grid_B2, grid2);
    B2->initialize();

    if(rank == 16) cout<<"Tensor B2 initialized"<<endl;

    int* size_C2 = new int[4];
    int* block_grid_C2 = new int[4];
    int* idmap_C2 = new int[4];

    size_C2[0] = size;
    size_C2[1] = size;
    size_C2[2] = size;
    size_C2[3] = size;

    block_grid_C2[0] = block_grid;
    block_grid_C2[1] = block_grid;
    block_grid_C2[2] = block_grid;
    block_grid_C2[3] = block_grid;

    idmap_C2[0] = 0;
    idmap_C2[1] = 1;
    idmap_C2[2] = 2;
    idmap_C2[3] = 3;

    Tensor* C2 = new Tensor("cccc", idmap_C2, size_C2, block_grid_C2, grid2);
    C2->initialize();

    if(rank == 16) cout<<"Tensor C2 initialized"<<endl;

    //double time =0.0;
    if(rank==16) cout<<"C [a,b,m,n ] = A [a,b,k,l ] x B [m,n,k,l ]"<<endl;
    Contraction* C02 = new Contraction(A2, B2, C2, grid2);
    time =-MPI_Wtime();
    C02->contract( "a,b,k,l", "m,n,k,l", "a,b,m,n");
	//}
	*/
	   MPI_Finalize();
	

}
