/*
We assume a 3x3 (radius: 1) convolution kernel, which is not separable.
Each work-group will process a (TILE_X x TILE_Y) tile of the image.
For coalescing, TILE_X should be multiple of 16.

Instead of examining the image border for each kernel, we recommend to pad the image
to be the multiple of the given tile-size.
*/

//should be multiple of 32 on Fermi and 16 on pre-Fermi...
#define TILE_X 32 

#define TILE_Y 8

// d_Dst is the convolution of d_Src with the kernel c_Kernel
// c_Kernel is assumed to be a float[11] array of the 3x3 convolution constants, one multiplier (for normalization) and an offset (in this order!)
// With & Height are the image dimensions (should be multiple of the tile size)
__kernel __attribute__( ( reqd_work_group_size( TILE_X, TILE_Y, 1 ) ) )
void Convolution(
				__global float* d_Dst,
				__global const float* d_Src,
				__constant float* c_Kernel,
				uint Width,  // Use width to check for image bounds
				uint Height,
				uint Pitch   // Use pitch for offsetting between lines
				)
{
	// OpenCL allows to allocate the local memory from 'inside' the kernel (without using the clSetKernelArg() call)
	// in a similar way to standard C.
	// the size of the local memory necessary for the convolution is the tile size + the halo area
	__local float tile[TILE_Y + 2 ][TILE_X + 2 ];

	// TO DO...
	

	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	
	int2 LID;
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);
	
	
	// Fill the halo with zeros
	if(LID.x==0 )
	{	
		tile[LID.y+1][LID.x]=0;
	}
	if(LID.x==TILE_X-1)
	{
		tile[LID.y+1][LID.x+2]=0;
	}
	if(LID.y==0)
	{
		tile[LID.y][LID.x]=0;
		if(LID.x==TILE_X-1)
		{
			tile[LID.y][LID.x+1]=0;
			tile[LID.y][LID.x+2]=0;
		}
	}
	if(LID.y==TILE_Y-1)
	{
		tile[LID.y+2][LID.x]=0;
		if(LID.x==TILE_X-1)
		{
			tile[LID.y+2][LID.x+1]=0;
			tile[LID.y+2][LID.x+2]=0;
		}
	}
	tile[LID.y+1][LID.x+1]=0;
barrier(CLK_LOCAL_MEM_FENCE);



	// Load main filtered area from d_Src
	if( GID.y < Height  && GID.x < Width )
	{
		tile[ LID.y + 1 ][ LID.x + 1 ] = d_Src[ GID.y * Pitch + GID.x ] ;
	}
	
	// Load halo regions from d_Src (edges and corners separately), check for image bounds!
	
	if( LID.x == 0 && GID.x > 0 )
	{
		tile[ LID.y + 1 ][ LID.x ] = d_Src[ GID.y * Pitch + ( GID.x - 1 ) ] ;
	
	}
	if( LID.x == ( TILE_X - 1 ) && GID.x < ( Width - 1 ) )
	{
		tile[ LID.y + 1 ][ LID.x + 2 ] = d_Src[ GID.y * Pitch + ( GID.x + 1 ) ] ;
	
	}
	if( LID.y == 0 && GID.y > 0 ) 
	{
		tile[ LID.y ][ LID.x + 1 ] = d_Src[ ( GID.y - 1 ) * Pitch + GID.x ] ;
	}
	if( LID.y == ( TILE_Y - 1 ) && GID.y < ( Height - 1 ) )
	{
		tile[ LID.y + 2 ][ LID.x + 1 ] = d_Src[ ( GID.y + 1 ) * Pitch + GID.x ]  ;
	}
	//coner
	if( LID.x == 0 && LID.y == 0 && GID.x > 0 && GID.y > 0 )
	{
		tile[ LID.y ][ LID.x ] = d_Src[ ( GID.y - 1 ) * Pitch + ( GID.x - 1 ) ] ;
	}
	if( LID.x == ( TILE_X - 1 ) && LID.y == 0 && GID.x < ( Width - 1 ) && GID.y > 0 )
	{
		tile[ LID.y ][ LID.x + 2 ] = d_Src[ ( GID.y - 1 ) * Pitch + ( GID.x + 1 ) ] ;
	}
	if( LID.x == 0 && LID.y == ( TILE_Y - 1 ) && GID.x > 0 && GID.y < ( Height - 1 ) )
	{
		tile[ LID.y + 2 ][ LID.x ] = d_Src[ ( GID.y + 1 ) * Pitch + ( GID.x - 1 ) ] ;
	}
	if( LID.x == ( TILE_X - 1 ) && LID.y == ( TILE_Y - 1 ) && GID.x < ( Width - 1 ) && GID.y < ( Height - 1 ) )
	{
		tile[ LID.y + 2 ][ LID.x + 2 ] = d_Src[ ( GID.y + 1 ) * Pitch + ( GID.x + 1 ) ] ;
	}
	
	// Sync threads
	
	barrier(CLK_LOCAL_MEM_FENCE);

	// Perform the convolution and store the convolved signal to d_Dst.
	float value = 0 ;
	for(int offsetY = -1; offsetY < 2; offsetY ++)
	{
		int sy = LID.y + 1 + offsetY ;
			for(int offsetX = -1; offsetX < 2; offsetX++)
			{
				int sx = LID.x + 1 + offsetX;
				value += tile[ sy ][ sx ] * c_Kernel[ ( 1 + offsetY ) * 3 + ( 1 + offsetX ) ];
			}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if( GID.y < Height  && GID.x < Width )
		d_Dst[ GID.y * Pitch + GID.x ] = value * c_Kernel[9] + c_Kernel[ 10 ] ;
	
}