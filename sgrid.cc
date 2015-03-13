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

	int size = 128;
    int block_grid = 8;
	
    int grid_dim = 4;
	
    int* pgrid = new int[grid_dim];
	//Grid Layout
	pgrid[0] = 2;
	pgrid[1] = 2;
	pgrid[2] = 4;
	pgrid[3] = 4;
  
	
	int ngrids=1;
	int* grid_sizes = new int[ngrids];
	grid_sizes[0]=np;
	
	Grid* grid = new Grid(grid_dim, pgrid,MPI_COMM_WORLD,0,ngrids,grid_sizes);
	
	int* size_A = new int[4];
    int* block_grid_A = new int[4];
    int* idmap_A = new int[4];

    size_A[0] = size;
    size_A[1] = size;
    size_A[2] = 8*size;
    size_A[3] = 8*size;

    block_grid_A[0] = block_grid;
    block_grid_A[1] = block_grid;
    block_grid_A[2] = block_grid;
    block_grid_A[3] = block_grid;

    idmap_A[0] = 0;
    idmap_A[1] = 1;
    idmap_A[2] = 2;
    idmap_A[3] = 3;


    int* size_B = new int[4];
    int* block_grid_B = new int[4];
    int* idmap_B = new int[4];

    size_B[0] = size;
    size_B[1] = size;
    size_B[2] = 8*size;
    size_B[3] = 8*size;

    block_grid_B[0] = block_grid;
    block_grid_B[1] = block_grid;
    block_grid_B[2] = block_grid;
    block_grid_B[3] = block_grid;

    idmap_B[0] = 2;
    idmap_B[1] = 3;
    idmap_B[2] = 0;
    idmap_B[3] = 1;

	
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

	
	int* size_A2 = new int[4];
    int* block_grid_A2 = new int[4];
    int* idmap_A2 = new int[4];

    size_A2[0] = size;
    size_A2[1] = size;
    size_A2[2] = 8*size;
    size_A2[3] = 8*size;

    block_grid_A2[0] = block_grid;
    block_grid_A2[1] = block_grid;
    block_grid_A2[2] = block_grid;
    block_grid_A2[3] = block_grid;

    idmap_A2[0] = 0;
    idmap_A2[1] = 1;
    idmap_A2[2] = 2;
    idmap_A2[3] = 3;

	
    int* size_B2 = new int[4];
    int* block_grid_B2 = new int[4];
    int* idmap_B2 = new int[4];

    size_B2[0] = size;
    size_B2[1] = size;
    size_B2[2] = 8*size;
    size_B2[3] = 8*size;

    block_grid_B2[0] = block_grid;
    block_grid_B2[1] = block_grid;
    block_grid_B2[2] = block_grid;
    block_grid_B2[3] = block_grid;

    idmap_B2[0] = 2;
    idmap_B2[1] = 3;
    idmap_B2[2] = 0;
    idmap_B2[3] = 1;


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
	
	int* size_A3 = new int[4];
    int* block_grid_A3 = new int[4];
    int* idmap_A3 = new int[4];

    size_A3[0] = size;
    size_A3[1] = size;
    size_A3[2] = 8*size;
    size_A3[3] = 8*size;

    block_grid_A3[0] = block_grid;
    block_grid_A3[1] = block_grid;
    block_grid_A3[2] = block_grid;
    block_grid_A3[3] = block_grid;

    idmap_A3[0] = 0;
    idmap_A3[1] = 1;
    idmap_A3[2] = 2;
    idmap_A3[3] = 3;


    int* size_B3 = new int[4];
    int* block_grid_B3 = new int[4];
    int* idmap_B3 = new int[4];

    size_B3[0] = size;
    size_B3[1] = size;
    size_B3[2] = 8*size;
    size_B3[3] = 8*size;

    block_grid_B3[0] = block_grid;
    block_grid_B3[1] = block_grid;
    block_grid_B3[2] = block_grid;
    block_grid_B3[3] = block_grid;

    idmap_B3[0] = 2;
    idmap_B3[1] = 3;
    idmap_B3[2] = 0;
    idmap_B3[3] = 1;

	
    int* size_C3 = new int[4];
    int* block_grid_C3 = new int[4];
    int* idmap_C3 = new int[4];

    size_C3[0] = size;
    size_C3[1] = size;
    size_C3[2] = size;
    size_C3[3] = size;

    block_grid_C3[0] = block_grid;
    block_grid_C3[1] = block_grid;
    block_grid_C3[2] = block_grid;
    block_grid_C3[3] = block_grid;

    idmap_C3[0] = 0;
    idmap_C3[1] = 1;
    idmap_C3[2] = 2;
    idmap_C3[3] = 3;

	
	int* size_A4 = new int[4];
    int* block_grid_A4 = new int[4];
    int* idmap_A4 = new int[4];

    size_A4[0] = size;
    size_A4[1] = size;
    size_A4[2] = 8*size;
    size_A4[3] = 8*size;

    block_grid_A4[0] = block_grid;
    block_grid_A4[1] = block_grid;
    block_grid_A4[2] = block_grid;
    block_grid_A4[3] = block_grid;

    idmap_A4[0] = 0;
    idmap_A4[1] = 1;
    idmap_A4[2] = 2;
    idmap_A4[3] = 3;

	
    int* size_B4 = new int[4];
    int* block_grid_B4 = new int[4];
    int* idmap_B4 = new int[4];

    size_B4[0] = size;
    size_B4[1] = size;
    size_B4[2] = 8*size;
    size_B4[3] = 8*size;

    block_grid_B4[0] = block_grid;
    block_grid_B4[1] = block_grid;
    block_grid_B4[2] = block_grid;
    block_grid_B4[3] = block_grid;

    idmap_B4[0] = 2;
    idmap_B4[1] = 3;
    idmap_B4[2] = 0;
    idmap_B4[3] = 1;


    int* size_C4 = new int[4];
    int* block_grid_C4 = new int[4];
    int* idmap_C4 = new int[4];

    size_C4[0] = size;
    size_C4[1] = size;
    size_C4[2] = size;
    size_C4[3] = size;

    block_grid_C4[0] = block_grid;
    block_grid_C4[1] = block_grid;
    block_grid_C4[2] = block_grid;
    block_grid_C4[3] = block_grid;

    idmap_C4[0] = 0;
    idmap_C4[1] = 1;
    idmap_C4[2] = 2;
    idmap_C4[3] = 3;
	
	
	
	Tensor* A = new Tensor("cccc", idmap_A, size_A, block_grid_A, grid);
    A->initialize();
    if(rank == 0) cout<<"Tensor A initialized"<<endl;
	
    Tensor* B = new Tensor("cccc", idmap_B, size_B, block_grid_B, grid);
    B->initialize();
    if(rank == 0) cout<<"Tensor B initialized"<<endl;
	
    Tensor* C = new Tensor("cccc", idmap_C, size_C, block_grid_C, grid);
    C->initialize();
    if(rank == 0) cout<<"Tensor C initialized"<<endl;
	
	Tensor* A2 = new Tensor("cccc", idmap_A2, size_A2, block_grid_A2, grid);
    A2->initialize();
    if(rank == 0) cout<<"Tensor A2 initialized"<<endl;
	
    Tensor* B2 = new Tensor("cccc", idmap_B2, size_B2, block_grid_B2, grid);
    B2->initialize();
    if(rank == 0) cout<<"Tensor B2 initialized"<<endl;
	
    Tensor* C2 = new Tensor("cccc", idmap_C2, size_C2, block_grid_C2, grid);
    C2->initialize();
    if(rank == 0) cout<<"Tensor C2 initialized"<<endl;

	Tensor* A3 = new Tensor("cccc", idmap_A3, size_A3, block_grid_A3, grid);
    A3->initialize();
    if(rank == 0) cout<<"Tensor A3 initialized"<<endl;
	
    Tensor* B3 = new Tensor("cccc", idmap_B3, size_B3, block_grid_B3, grid);
    B3->initialize();
    if(rank == 0) cout<<"Tensor B3 initialized"<<endl;
	
    Tensor* C3 = new Tensor("cccc", idmap_C3, size_C3, block_grid_C3, grid);
    C3->initialize();
    if(rank == 0) cout<<"Tensor C initialized"<<endl;
	
	Tensor* A4 = new Tensor("cccc", idmap_A4, size_A4, block_grid_A4, grid);
    A4->initialize();
    if(rank == 0) cout<<"Tensor A4 initialized"<<endl;
	
    Tensor* B4 = new Tensor("cccc", idmap_B4, size_B4, block_grid_B4, grid);
    B4->initialize();
    if(rank == 0) cout<<"Tensor B4 initialized"<<endl;
	
    Tensor* C4 = new Tensor("cccc", idmap_C4, size_C4, block_grid_C4, grid);
    C4->initialize();
    if(rank == 0) cout<<"Tensor C4 initialized"<<endl;
	
	double time =0.0,max_time=0.0;
	time =-MPI_Wtime();
	
    if(rank==0) cout<<"C [a,b,m,n ] = A [a,b,k,l ] x B [m,n,k,l ]"<<endl;
    Contraction* C0 = new Contraction(A, B, C, grid);
    C0->contract( "a,b,k,l", "m,n,k,l", "a,b,m,n");
	if(rank==0) cout<<"C done"<<endl;
	
    if(rank==0) cout<<"C2 [a,b,m,n ] = A [a,b,k,l ] x B [m,n,k,l ]"<<endl;
    Contraction* C02 = new Contraction(A2, B2, C2, grid);
    C02->contract( "a,b,k,l", "m,n,k,l", "a,b,m,n");
	if(rank==0) cout<<"C2 done"<<endl;
	
	if(rank==0) cout<<"C3 [a,b,m,n ] = A [a,b,k,l ] x B [m,n,k,l ]"<<endl;
    Contraction* C03 = new Contraction(A3, B3, C3, grid);
    C03->contract( "a,b,k,l", "m,n,k,l", "a,b,m,n");
	if(rank==0) cout<<"C3 done"<<endl;
	
	if(rank==0) cout<<"C4 [a,b,m,n ] = A [a,b,k,l ] x B [m,n,k,l ]"<<endl;
    Contraction* C04 = new Contraction(A4, B4, C4, grid);
    C04->contract( "a,b,k,l", "m,n,k,l", "a,b,m,n");
	if(rank==0) cout<<"C4 done"<<endl;
	
	time +=MPI_Wtime();
	MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(rank==0) cout<<"Time: "<<max_time<<endl;
	
	MPI_Finalize();
	

}
