
__kernel void
set_array_to_constant(
	__global int *array,
	int num_elements,
	int val
)
{
	// There is no need to touch this kernel
	if(get_global_id(0) < num_elements)
		array[get_global_id(0)] = val;
}

__kernel void
compute_histogram(
	__global int *histogram,   // accumulate histogram here
	__global const float *img, // input image
	int width,                 // image width
	int height,                // image height
	int pitch,                 // image pitch
	int num_hist_bins          // number of histogram bins
)
{
	// Insert your kernel code here
	
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);	
	if( ( GID.y < height ) && ( GID.x < width ) )
	{
		float p = img[GID.y * pitch + GID.x] * (float)(num_hist_bins);
		int h_idx = (int)min(num_hist_bins - 1, (int)max(0, (int)(p)));
		atomic_add( &histogram[h_idx] , 1 ) ;
	}
	
} 

__kernel void
compute_histogram_local_memory(
	__global int *histogram,   // accumulate histogram here
	__global const float *img, // input image
	int width,                 // image width
	int height,                // image height
	int pitch,                 // image pitch
	int num_hist_bins,         // number of histogram bins
	__local int *local_hist
)
{
	// Insert your kernel code here
	
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	
	int2 LID;
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);
	
	int2 GPID;
	GPID.x = get_group_id(0);
	GPID.y = get_group_id(1);
	
	if( ( LID.y * get_local_size(0) + LID.x ) < num_hist_bins )
	{
		local_hist[ LID.y * get_local_size(0) + LID.x ] = 0 ;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if( ( GID.y < height ) && ( GID.x < width ) )
	{
		float p = img[GID.y * pitch + GID.x] * (float)(num_hist_bins);
		int h_idx = (int)min(num_hist_bins - 1, (int)max(0, (int)(p)));
		atomic_add( &local_hist[h_idx] , 1 ) ;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if( ( LID.y * get_local_size(0) + LID.x ) < num_hist_bins )
	{
		int local_value = local_hist[LID.y * get_local_size(0) + LID.x];
		atomic_add( &histogram[LID.y * get_local_size(0) + LID.x] , local_value ) ;
	
	}
	
	
	
} 
