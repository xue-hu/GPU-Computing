
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_InterleavedAddressing(__global uint* array, uint stride) 
{
	// TO DO: Kernel implementation
	int GID = get_global_id(0);
	int ldx = GID * 2 * stride ;
	int rdx = (2*GID+1) * stride  ;
	array[ ldx ] = array[ ldx ] + array[ rdx ] ;
	
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_SequentialAddressing(__global uint* array, uint stride) 
{
	// TO DO: Kernel implementation
	int GID = get_global_id(0);
	int ldx = GID  ;
	int rdx = GID + stride  ;
	array[ ldx ] = array[ ldx ] + array[ rdx ] ;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_Decomp(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localBlock)
{
	// TO DO: Kernel implementation
	
	int GID = get_global_id(0);
	int LID = get_local_id(0);
	int GPID = get_group_id(0);
	
	int ldx = GPID * get_local_size(0) * 2 + LID ;
	int rdx = GPID * get_local_size(0) * 2 + LID + get_local_size(0) ;
	if( rdx < N )
	{
		localBlock[LID] = inArray[ ldx ] + inArray[ rdx ] ;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(unsigned int stride = 1 ; stride <= get_local_size(0)/2 ; stride=stride*2 ) 
	{
		ldx = LID * 2 * stride;
		rdx = ( 2 * LID + 1 ) * stride ;
		if( rdx < get_local_size(0) )
		{
			localBlock[ldx] = localBlock[ ldx ] + localBlock[ rdx ] ;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
		
	if( LID == 0 )
	{
		outArray[ GPID ] = localBlock[0] ;
	
	}
	
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_DecompUnroll(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localBlock)
{
	// TO DO: Kernel implementation
}
