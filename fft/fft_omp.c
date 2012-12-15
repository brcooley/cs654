//-lfftw3_omp -lfftw3 -lm
/* cc -I$HOME/usr/include simple_example.c -L$HOME/usr/lib -lfftw3 -o simple_example */
 
 #include <fftw3.h>
 #include <math.h>
 #include <omp.h> 
     
 int main(int argc, char **argv){
   int fftw_init_threads(void); //before calling any fftw routine
    omp_set_num_threads(8) //before the thing below it
   const ptrdiff_t N0 = 18, N1 = 18;
   fftw_plan plan;
   fftw_complex *data;
 
   data = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N0 * N1);
 
   fftw_plan_with_nthreads(omp_get_max_threads()) //before creating a plan
   /* create plan for forward DFT */
   plan = fftw_plan_dft_2d(N0, N1, data, data, FFTW_FORWARD, FFTW_ESTIMATE);
 
   /* initialize data to some function my_function(x,y) */
   int i, j;
   double pdata=0;
   for (i = 0; i < N0; ++i){
     for (j = 0; j < N1; ++j){
       data[i*N1 + j][0]=i; 
       data[i*N1 + j][1]=0;
       pdata+=data[i*N1 + j][0]*data[i*N1 + j][0]+data[i*N1 + j][1]*data[i*N1 + j][1];
     }
   }
   printf("power of original data is %f\n", pdata);
 
   /* compute transforms, in-place, as many times as desired */
   fftw_execute(plan);
 
   double normalization=sqrt((double)N0*N1);
   double ptransform = 0;
 
   /*normalize data and calculate power of transform */
   for (i = 0; i < N0; ++i){
     for (j = 0; j < N1; ++j){
       data[i*N1+j][0]/=normalization;
       data[i*N1+j][1]/=normalization;
       ptransform+=data[i*N1 + j][0]*data[i*N1 + j][0]+data[i*N1 + j][1]*data[i*N1 + j][1];
     }
   }
 
   printf("power of transform is %f\n", pdata);
  
   fftw_destroy_plan(plan);
   fftw_free(data); 
   return 0;
 }
