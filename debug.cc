#include "mpi.h"
#include "grid.h"
#include "tensor.h"
#include "contraction.h"
#include "redistribute.h"
#include "grid_redib.h"
#include "helper.h"
using namespace RRR;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank,np;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    MPI_Comm_rank(MPI_COMM_WORLD,&np);

    int size = 16;
    int block_grid = 4;
    int grid_dim = 4;
    int* pgrid1 = new int[grid_dim];
 
	pgrid1[0] = 2;
	pgrid1[1] = 2;
	pgrid1[2] = 2;
	pgrid1[3] = 2;
    
	int Vb=32, Va=32, Oa=4, Ob=4;
   

    Grid* grid1 = new Grid(grid_dim, pgrid1,MPI_COMM_WORLD);
	
	int grid_dim2=2;
	int* pgrid2=new int[grid_dim2];
	pgrid2[0]=4;
	pgrid2[1]=4;
	
	Grid* grid2 = new Grid(grid_dim2, pgrid2,MPI_COMM_WORLD);

    int* size_tb_vo = new int[2];
    size_tb_vo[0] = Vb;
    size_tb_vo[1] = Ob;
    int* idmap_tb_vo = new int[2];
    idmap_tb_vo[0] = 0;
    idmap_tb_vo[1] = 1;
    int* vgrid_tb_vo = new int[2];
    vgrid_tb_vo[0] = 4;
    vgrid_tb_vo[1] = 4;
    Tensor* tb_vo = new Tensor("cc", idmap_tb_vo, size_tb_vo, vgrid_tb_vo, grid1);
    tb_vo->initialize();
   

    if(rank == 0) cout<<"Tensor vab_oovv initialized"<<endl;

	int* idmap_tb_vo2 = new int[2];
    idmap_tb_vo2[0] = 3;
    idmap_tb_vo2[1] = 2;
  

	

	    if(rank == 15) cout << "Redistributing "<< endl;


	    //time=-MPI_Wtime();
GridRedistribute* redib1 = new GridRedistribute(tb_vo,idmap_tb_vo2,grid1);
	redib1->redistribute();


int* size_tb_vob = new int[2];
    size_tb_vob[0] = Vb;
    size_tb_vob[1] = Ob;
    int* idmap_tb_vob = new int[2];
    idmap_tb_vob[0] = 3;
    idmap_tb_vob[1] = 2;
    int* vgrid_tb_vob = new int[2];
    vgrid_tb_vob[0] = 4;
    vgrid_tb_vob[1] = 4;
    Tensor* tb_vob = new Tensor("cc", idmap_tb_vob, size_tb_vob, vgrid_tb_vob, grid1);
    tb_vob->initialize();
   

    if(rank == 0) cout<<"Tensor vab_oovv initialized"<<endl;


	cout<<"Rank: "<<tb_vo->rank<<"tb_vob->num_actual_tiles"<<tb_vob->num_actual_tiles<<"	tb_vo->num_actual_tiles"<<tb_vo->num_actual_tiles<<endl;
	are_tile_addrs_equal(2, tb_vo->tile_address, tb_vob->tile_address, tb_vob->num_actual_tiles);


	    MPI_Barrier(MPI_COMM_WORLD);
	    if (rank == 15) cout << "Redistribution  took : "<<time<<" seconds"<<endl;
	   
	

}
