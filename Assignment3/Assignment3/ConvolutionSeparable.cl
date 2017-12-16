
//Each thread load exactly one halo pixel
//Thus, we assume that the halo size is not larger than the 
//dimension of the work-group in the direction of the kernel

//to efficiently reduce the memory transfer overhead of the global memory
// (each pixel is lodaded multiple times at high overlaps)
// one work-item will compute RESULT_STEPS pixels

//for unrolling loops, these values have to be known at compile time

/* These macros will be defined dynamically during building the program

#define KERNEL_RADIUS 2

//horizontal kernel
#define H_GROUPSIZE_X		32
#define H_GROUPSIZE_Y		8
#define H_RESULT_STEPS		2

//vertical kernel
#define V_GROUPSIZE_X		8
#define V_GROUPSIZE_Y		32
#define V_RESULT_STEPS		3

*/


#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)


//////////////////////////////////////////////////////////////////////////////////////////////////////
// Horizontal convolution filter

/*
c_Kernel stores 2 * KERNEL_RADIUS + 1 weights, use these during the convolution
*/

//require matching work-group size
__kernel __attribute__((reqd_work_group_size(H_GROUPSIZE_X, H_GROUPSIZE_Y, 1)))
void ConvHorizontal(
			__global float* d_Dst,
			__global const float* d_Src,
			__constant float* c_Kernel,
			int Width,
			int Pitch
			)
{
	//The size of the local memory: one value for each work-item.
	//We even load unused pixels to the halo area, to keep the code and local memory access simple.
	//Since these loads are coalesced, they introduce no overhead, except for slightly redundant local memory allocation.
	//Each work-item loads H_RESULT_STEPS values + 2 halo values
	
	__local float tile[H_GROUPSIZE_Y][(H_RESULT_STEPS + 2) * H_GROUPSIZE_X];

	// TODO:
	
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	
	int2 LID;
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);
	
	int2 GPID;
	GPID.x = get_group_id(0);
	GPID.y = get_group_id(1);
	
	const int baseX = GPID.x * H_RESULT_STEPS * H_GROUPSIZE_X ;
	const int baseY = GID.y ;
	const int offset = baseY * Pitch + baseX ;

	// Load left halo (check for left bound)
	if(( baseX + LID.x - H_GROUPSIZE_X ) >= 0 )
	{
		tile[ LID.y ][ LID.x ] = d_Src[ offset + LID.x - H_GROUPSIZE_X ] ;
	}
	else
	{
		tile[ LID.y ][ LID.x ] = 0 ;
	}
	
	// Load main data + right halo (check for right bound)
	for (int tileID = 1; tileID <= H_RESULT_STEPS ; tileID++ )
	{
		tile[ LID.y ][ tileID * H_GROUPSIZE_X + LID.x ] = d_Src[ offset + LID.x + ( tileID - 1 ) * H_GROUPSIZE_X ] ;
	}

	if(( baseX + LID.x + H_GROUPSIZE_X * H_RESULT_STEPS ) < Width )
	{
		tile[ LID.y ][ LID.x + H_GROUPSIZE_X * ( H_RESULT_STEPS + 1 ) ] = d_Src[ offset + LID.x + H_RESULT_STEPS * H_GROUPSIZE_X ] ;
	}
	else
	{
		tile[ LID.y ][ LID.x + H_GROUPSIZE_X * ( H_RESULT_STEPS + 1 ) ] = 0 ;
	}
	
	
	// Sync the work-items after loading
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Convolve and store the result
	
	for (int tileID = 1; tileID <=H_RESULT_STEPS ; tileID++ )
		{
			float value = 0;
			//apply horizontal kernel
			for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
			{
				int sx = tileID * H_GROUPSIZE_X + LID.x + k;
				if(sx >= 0 && sx < (H_RESULT_STEPS + 2) * H_GROUPSIZE_X)
					value += tile[LID.y][sx] * c_Kernel[KERNEL_RADIUS - k];
			}
			d_Dst[offset + LID.x + ( tileID - 1 ) * H_GROUPSIZE_X] = value ;
		
		
		}

}

//////////////////////////////////////////////////////////////////////////////////////////////////////
// Vertical convolution filter

//require matching work-group size
__kernel __attribute__((reqd_work_group_size(V_GROUPSIZE_X, V_GROUPSIZE_Y, 1)))
void ConvVertical(
			__global float* d_Dst,
			__global const float* d_Src,
			__constant float* c_Kernel,
			int Height,
			int Pitch
			)
{
	__local float tile[(V_RESULT_STEPS + 2) * V_GROUPSIZE_Y][V_GROUPSIZE_X];

	//TO DO:
	
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	
	int2 LID;
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);
	
	int2 GPID;
	GPID.x = get_group_id(0);
	GPID.y = get_group_id(1);
	
	const int baseX = GID.x ;
	const int baseY = GPID.y * H_RESULT_STEPS * V_GROUPSIZE_Y ;
	
	// Conceptually similar to ConvHorizontal
	// Load top halo 
	
	if(( baseY + LID.y - V_GROUPSIZE_Y ) >= 0 )
	{
		tile[ LID.y ][ LID.x ] = d_Src[ ( baseY + LID.y - V_GROUPSIZE_Y ) * Pitch + GID.x ] ;
	}
	else
	{
		tile[ LID.y ][ LID.x ] = 0 ;
	}
	
	// + main data + bottom halo
	
	for (int tileID = 1; tileID <= H_RESULT_STEPS ; tileID++ )
	{
		tile[ tileID * V_GROUPSIZE_Y + LID.y ][ LID.x ] = d_Src[ ( baseY + ( tileID - 1 ) * V_GROUPSIZE_Y + LID.y ) * Pitch + GID.x ] ;
	}

	if(( baseY + LID.y + V_GROUPSIZE_Y * H_RESULT_STEPS ) < Height )
	{
		tile[ LID.y + V_GROUPSIZE_Y * ( H_RESULT_STEPS + 1 ) ][ LID.x ] = d_Src[ ( baseY + H_RESULT_STEPS * V_GROUPSIZE_Y + LID.y ) * Pitch + GID.x ] ;
	}
	else
	{
		tile[ LID.y + V_GROUPSIZE_Y * ( H_RESULT_STEPS + 1 ) ][ LID.x ] = 0 ;
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	
	// Compute and store results

	for (int tileID = 1; tileID <=H_RESULT_STEPS ; tileID++ )
		{
			float value = 0;
			//apply horizontal kernel
			for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
			{
				int sy = tileID * V_GROUPSIZE_Y + LID.y + k;
				if(sy >= 0 && sy < (H_RESULT_STEPS + 2) * V_GROUPSIZE_Y )
					value += tile[sy][LID.x] * c_Kernel[KERNEL_RADIUS - k];
			}
			d_Dst[ ( baseY + ( tileID - 1 ) * V_GROUPSIZE_Y + LID.y ) * Pitch + GID.x] = value ; 
		
		}
}
