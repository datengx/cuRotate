#ifndef __ITKIO_H__
#define __ITKIO_H__


#include <string.h>

/*
*	Define the mission for this file
*   ITK image is convenient to work with once you have created them. IT take some time to create them, which is a little
*	bit time consuming. The function defined in this file will help you to create a ITK image, copy the buffer that
*   that you have somewhere that represents an image, an does IO job for you.
*/


// template <typename Type>
void store_mha( float* buffer,
				const uint dim,
				const uint h,
				const uint w,
				const uint d,
				const char* filename
				 );


#endif