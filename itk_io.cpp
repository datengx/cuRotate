/* ITK include */
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImportImageFilter.h"
#include "itk_io.h"

const unsigned int Dim = 3;
typedef float                                  PixelType;

// template <typename Type>
void store_mha( float* buffer,
				const uint dim,
				const uint h,
				const uint w,
				const uint d,
				const char* filename
				 ) {


	typedef itk::Image< PixelType, Dim >  	      ImageType;
	typedef itk::ImageFileWriter< ImageType >     WriterType;
	typedef itk::ImportImageFilter< PixelType, Dim >   ImportFilterType;


	/* Create an image holder */
	ImportFilterType::SizeType  size;
	size[0] = h; size[1] = w; size[2] = d;

	ImportFilterType::IndexType start;
	start.Fill( 0 );
	ImportFilterType::Pointer importFilter = ImportFilterType::New();

	ImportFilterType::RegionType region;
	region.SetIndex( start );
	region.SetSize(  size  );

	PixelType origin[ Dim ];
	origin[0] = 0.0;    // X coordinate
	origin[1] = 0.0;    // Y coordinate
	origin[2] = 0.0;    // Z coordinate

	PixelType spacing[ Dim ];
	spacing[0] = 1.0;    // along X direction
	spacing[1] = 1.0;    // along Y direction
	spacing[2] = 1.0;    // along Z direction

	importFilter->SetRegion( region );
	importFilter->SetOrigin( origin );
	importFilter->SetSpacing( spacing );

	const unsigned long numberOfPixels =  size[0] * size[1] * size[2];
	const bool importImageFilterWillOwnTheBuffer = false;
	importFilter->SetImportPointer( buffer, numberOfPixels,
                                    importImageFilterWillOwnTheBuffer );

	importFilter->Update();

	WriterType::Pointer writer = WriterType::New();
    writer->SetFileName( filename );
    writer->SetInput( importFilter->GetOutput() );
    writer->Update();
}