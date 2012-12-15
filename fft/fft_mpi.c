#include <fftw3-mpi.h>
#include <sys/time.h>


double startTime, endTime;
/*
double getTime() {
       timeval thetime;
       gettimeofday( &thetime, 0 );
       return thetime.tv_sec + thetime.tv_usec / 1000000.0;
}
*/
int main(int argc, char **argv)
{
    const ptrdiff_t N0 = 1024, N1 = 1024;
    fftwf_plan plan;
    fftwf_complex *data;
    ptrdiff_t alloc_local, local_n0, local_0_start, i, j;



    MPI_Init(&argc, &argv);
    fftwf_mpi_init();

    /* get local data size and allocate */
    alloc_local = fftwf_mpi_local_size_2d(N0, N1, MPI_COMM_WORLD,
                                         &local_n0, &local_0_start);
    data = fftwf_alloc_complex(alloc_local);//(fftwf_complex *) fftwf_malloc(sizeof(fftw_complex) * alloc_local);

    /* create plan for in-place forward DFT */
    plan = fftwf_mpi_plan_dft_2d(N0, N1, data, data, MPI_COMM_WORLD,
                                FFTW_FORWARD, FFTW_ESTIMATE);

    /* initialize data to some function my_function(x,y) */
    for (i = 0; i < local_n0; ++i) for (j = 0; j < N1; ++j){
        data[i*N1 + j][0] = local_0_start;;//my_function(local_0_start + i, j);
        data[i*N1 + j][1]=i;
    }

    /* compute transforms, in-place, as many times as desired */
    
    //startTime = getTime();
    fftwf_execute(plan);
    //endTime= getTime();

    fftwf_destroy_plan(plan);
    fftwf_mpi_cleanup();
    MPI_Finalize();

    printf("\n%.5lf\n",endTime-startTime);

    return 0;
}
