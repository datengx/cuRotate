
#include "../common/book.h"
#include "./utils.h"
#include "./timing.h"
// #include "./cuda_kernels.cuh"
#pragma clang diagnostic push
#pragma ide diagnostic ignored "CannotResolve"
#include <cuda.h>
#include <cufft.h>
#include <iostream>
#include <complex>
#include <vector>
#include <cstring>


#define NX 128
#define NY 128
#define NZ 128
#define LX (2 * M_PI)
#define LY (2 * M_PI)
#define NUM_IMAGES 3


using namespace std;

typedef double     SimPixelType;

// __global__ void add_slices(PIXEL_TYPE* image_in, PIXEL_TYPE* image_out) {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     int idx = tid % (128 * 128 * 2);
//     PIXEL_TYPE temp = image_in[tid];
//     // printf("%d\n", idx);
//     // if (tid < TOTAL_PIXELS) {
//     // if (tid < 16384) {
//     atomicAdd( &image_out[idx], temp );
//         // image_out[tid] = temp;
//     // }
//     // }
// }

 __global__ void Multiply_complex(SimPixelType* image_in, SimPixelType* image_in2) {
     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     int idx = tid % (128 * 128 * 2);
     SimPixelType c1_real = image_in[tid*2];
	 SimPixelType c1_imag = image_in[tid*2+1];
	 SimPixelType c2_real = image_in2[tid*2];
	 SimPixelType c2_imag = image_in2[tid*2+1];
	 image_in[tid*2] = c1_real * c2_real - c1_imag * c2_imag;
	 image_in[tid*2+1] = c1_real * c2_imag + c1_imag * c2_real;
 }

int main() {
	cudaDeviceReset();
	/* Create couple of images for testing */
	SimPixelType *x = new SimPixelType[NX * NY];
	SimPixelType *y = new SimPixelType[NX * NY];
	SimPixelType *in = new SimPixelType[NX * NY * NZ];
	/* A vector holding multiple images data */
	vector< SimPixelType* > image_vector;
	vector< SimPixelType* > dev_pointers_in;
	vector< SimPixelType* > dev_pointers_out;
	vector< SimPixelType* > imageOut_vector;
	vector< SimPixelType* > mult_image_vector;

	/* Create Fourier Kernel plan */
	cufftHandle planr2c[NUM_IMAGES];
	cufftHandle planc2r[NUM_IMAGES];

	/* Create an array of CUDA streams */
	cudaStream_t streams_fft[NUM_IMAGES];

	/* Output image */
	complex<SimPixelType> *out = new complex<SimPixelType>[NX * NY * NZ];
	gpuErrchk( cudaHostRegister( out, sizeof(SimPixelType)*NX*NY*NZ*2, cudaHostRegisterPortable ) );
	// complex<SimPixelType>* out;
	// gpuErrchk( cudaMallocHost( &out, NX * NY * NZ * sizeof(SimPixelType) * 2 ) );
		/* Initialize it */
	memset( out, 0, sizeof(SimPixelType)*NX*NY*NZ*2 );

	/* Create the second argument image in the multiply kernel */
	SimPixelType* OTF = new SimPixelType[NX * NY * NZ * 2]; // Since the image is complex
	SimPixelType* dev_OTF;

	for (int p = 0; p < NZ; p++) {
		for(int j = 0; j < NY; j++) {
			for(int kk = 0; kk < NX; kk++) {
				OTF[(j * NX + kk) * 2] = kk + j;
				OTF[(j * NX + kk) * 2 + 1] = kk + j;
			}
		}
	}
	/* Reserve memory locations for the OTF image */
	gpuErrchk( cudaMalloc( &dev_OTF, sizeof(SimPixelType)*NX*NY*NZ*2 ) );
	gpuErrchk( cudaHostRegister( OTF, sizeof(SimPixelType)*NX*NY*NZ*2, cudaHostRegisterPortable ) );

	for (unsigned i = 0; i < NUM_IMAGES; i++) {

		SimPixelType *vx = new SimPixelType[NX * NY * NZ];
		SimPixelType *mult_image = new SimPixelType[NX * NY * NZ * 2];
		// SimPixelType* vx;
		// cudaMallocHost( &vx, NX * NY * NZ * sizeof(SimPixelType) );
		for (int p = 0; p < NZ; p++) {
			for(int j = 0; j < NY; j++){
			    for(int kk = 0; kk < NX; kk++){
			        x[j * NX + kk] = kk * LX/NX;
			        y[j * NX + kk] = kk * LY/NY;

			        /* Put values in the new images */
			        vx[j * NX + kk + p * NX * NY] = cos(x[j * NX + kk] + y[j * NX + kk]);
			        if ( i == 0 ) {
			        	in[j * NX + kk + p * NX * NY] = cos(x[j * NX + kk] + y[j * NX + kk]);
			        }
			    }
			}
		}
		t1 = absoluteTime();
		gpuErrchk( cudaHostRegister( vx, sizeof(SimPixelType)*NX*NY*NZ, cudaHostRegisterPortable ) );
		gpuErrchk( cudaHostRegister( mult_image, sizeof(SimPixelType)*NX*NY*NZ*2, cudaHostRegisterPortable ) );
		t2 = absoluteTime();
  		std::cout << "\n\n Register time: " << (float)(t2-t1)/1000000 << "ms" << std::endl;
		// for (int j = 0; j < NY; j++){
		//     for (int i = 0; i < NX; i++){
		//         // printf("%.3f ", vx[j*NX + i]/(NX*NY));
		//         cout << vx[j * NX + i] << " ";
		//     }
		//     // printf("\n");
		//     cout << endl;
		// }
		// cout << endl;
		/* Allocate some spaces on the device */
		SimPixelType *d_vx;
		SimPixelType *d_out;
		/* Some space on the device */
		gpuErrchk(cudaMalloc(&d_vx, NX * NY * NZ * sizeof(SimPixelType)));
		gpuErrchk(cudaMalloc(&d_out, NX * NY * NZ * sizeof(cufftDoubleComplex)));

		/* Create cufft FFT plans */
		int n[2] = {NX, NY};
		int inembed[] = {NX, NY};
		int onembed[] = {NX, NY};

		/* Forward Fourier Transform plan */
		cufftPlanMany(&planr2c[i],
		            2, // rank
		            n, // dimension
		            inembed,
		            1, // istride
		            NX * NY, // idist
		            onembed,
		            1, //ostride
		            NX * NY, // odist
		            CUFFT_D2Z,
		            NZ);



		/* Inverse Fourier Transform plan */
		cufftPlanMany(&planc2r[i],
		            2, // rank
		            n, // dimension
		            onembed,
		            1, // istride
		            NX * NY, // idist
		            inembed,
		            1, //ostride
		            NX * NY, // odist
		            CUFFT_Z2D,
		            NZ);

		cufftSetCompatibilityMode(planr2c[i], CUFFT_COMPATIBILITY_NATIVE);
		cufftSetCompatibilityMode(planc2r[i], CUFFT_COMPATIBILITY_NATIVE);
		/* Create streams associated with this 2 plans  */
		gpuErrchk( cudaStreamCreate( &streams_fft[i] ));
		cufftSetStream( planr2c[i], streams_fft[i] );
		// gpuErrchk( cudaStreamCreate(&streams_ifft[i]) );
		// cufftSetStream(&planc2r[i]);

		image_vector.push_back( vx );
		mult_image_vector.push_back( mult_image );
		dev_pointers_in.push_back( d_vx );
		dev_pointers_out.push_back( d_out );
	}


	/* Copying data to the device for processing */
	// cudaMemcpy(d_vx, vx, NX * NY * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);
	// cudaMemcpy(d_out, out, NX * NY * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
	t1 = absoluteTime();
	gpuErrchk( cudaMemcpyAsync(
				dev_OTF,
				OTF,
				2*NX*NY*NZ*sizeof(SimPixelType),
				cudaMemcpyHostToDevice,
				streams_fft[0]
	) );

	for (unsigned int j = 0; j < NUM_IMAGES; j++ ) {
		gpuErrchk( cudaMemcpyAsync( dev_pointers_in[j],
									image_vector[j],
									NX*NY*NZ*sizeof(SimPixelType),
									cudaMemcpyHostToDevice,
									streams_fft[j]) );
		gpuErrchk( cudaMemcpyAsync( dev_pointers_out[j],
									out,
									NX*NY*NZ*sizeof(cufftDoubleComplex),
									cudaMemcpyHostToDevice,
									streams_fft[j] ) );

	}

	for (unsigned int j = 0; j < NUM_IMAGES; j++) {
		cufftExecD2Z( planr2c[j],
					  (SimPixelType*)dev_pointers_in[j],
					  (cufftDoubleComplex*)dev_pointers_out[j]);
		Multiply_complex<<< NX*NY*NZ/512, 512, 0, streams_fft[j] >>>( dev_pointers_out[j],
						  dev_OTF
							);
	}

	t2 = absoluteTime();
  	std::cout << "\n\n Streaming time: " << (float)(t2-t1)/1000000 << "ms" << std::endl;
//	for (unsigned int j = 0; j < NUM_IMAGES; j++) {
//		cufftSetStream(planc2r[j], streams_fft[j]);
//	}
//
//	for (unsigned int j = 0; j < NUM_IMAGES; j++) {
//		cufftExecZ2D( planc2r[j], (cufftDoubleComplex*)dev_pointers_out[j], (SimPixelType*)dev_pointers_in[j]);
//	}

	for (unsigned int j = 0; j < NUM_IMAGES; j++) {
		gpuErrchk( cudaMemcpyAsync( mult_image_vector[j], dev_pointers_out[j], 2*NX*NY*NZ*sizeof(SimPixelType), cudaMemcpyDeviceToHost, streams_fft[j] ) );
	}

	for (unsigned int j = 0; j < NUM_IMAGES; j++) {
		gpuErrchk( cudaStreamSynchronize( streams_fft[j] ) );
	}
	t1 = absoluteTime();
	for (unsigned int j = 0; j < NUM_IMAGES; j++) {
		gpuErrchk( cudaHostUnregister(image_vector[j]) );
		gpuErrchk( cudaHostUnregister(mult_image_vector[j]) );
		// gpuErrchk( cudaFreeHost(image_vector[j]) );
	}
	gpuErrchk( cudaHostUnregister(OTF) );
	gpuErrchk( cudaHostUnregister(out) );
	// gpuErrchk( cudaFreeHost( out ) );
	t2 = absoluteTime();
  	std::cout << "\n\n Host Unregister time: " << (float)(t2-t1)/1000000 << "ms" << std::endl;

	/* Cast into complex value array */
	complex< SimPixelType >* complex_array = reinterpret_cast< complex< SimPixelType >* >( mult_image_vector[1] );

   	for (int j = 0; j < NY; j++){
	     for (int i = 0; i < NX; i++){
	         // printf("%.3f ", vx[j*NX + i]/(NX*NY));
	         // SimPixelType* vx = image_vector[1];
//	         cout << image_vector[0][j * NX + i]/( NX * NY ) << " ";
			cout << complex_array[j * NX + i] << " ";
	     }
	     // printf("\n");
	     cout << endl;
	 }
	// cout << endl;
	// for (int j = 0; j < NY; j++){
	//     for (int i = 0; i < NX; i++){
	//         // printf("%.3f ", vx[j*NX + i]/(NX*NY));
	//         cout << in[j * NX + i] << " ";
	//     }
	//     // printf("\n");
	//     cout << endl;
	// }

	for (unsigned int j = 0; j < NUM_IMAGES; j++) {
		gpuErrchk( cudaFree( dev_pointers_in[j] ) );
		gpuErrchk( cudaFree( dev_pointers_out[j] ) );
		cudaStreamDestroy( streams_fft[j] );
		delete[] image_vector[j];
		delete[] mult_image_vector[j];
	}
	gpuErrchk( cudaFree( dev_OTF ) );
	delete[] OTF;
	delete[] out;
	delete[] x;
	delete[] y;

	cudaDeviceReset();

	// cufftPlan2d(&planr2c, NY, NX, CUFFT_D2Z);
	// cufftPlan2d(&planc2r, NY, NX, CUFFT_Z2D);

	// cufftExecD2Z(planr2c, (cufftDoubleReal *)d_vx, (cufftDoubleComplex *)d_out);
	// cufftExecZ2D(planc2r, (cufftDoubleComplex *)d_out, (cufftDoubleReal *)d_vx);


	/* Copy results back from the device */
	// cudaMemcpy(vx, d_vx, NX * NY * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);







    // int count = 0;
    // cudaDeviceProp prop;
    // int dev_id;
    // //  determining how many devices are available to use on the computer
    // HANDLE_ERROR( cudaGetDeviceCount( &count ) );
    // printf("There are %d device(s) on this computer.\n", count);
    // // Iterates through each of the device on this computer

    // printDevInfo(count, prop);



	return 0;
}
#pragma clang diagnostic pop