


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_Naive(const __global uint* inArray, __global uint* outArray, uint N, uint offset) 
{
	// TO DO: Kernel implementation
	
	int GID = get_global_id(0);
	if( GID < N )
	{
		int ldx = GID - offset ;
		int rdx = GID ;
	
		if( ldx < 0 )
			outArray[ rdx ] = inArray[ rdx ] ;
		else
			outArray[ rdx ] = inArray[ ldx ] + inArray[ rdx ] ;
	}	
}



// Why did we not have conflicts in the Reduction? Because of the sequential addressing (here we use interleaved => we have conflicts).

#define UNROLL
#define NUM_BANKS			32
#define NUM_BANKS_LOG		5
#define SIMD_GROUP_SIZE		32

// Bank conflicts
#define AVOID_BANK_CONFLICTS
#ifdef AVOID_BANK_CONFLICTS
	// TO DO: define your conflict-free macro here
#else
	#define OFFSET(A) (A)
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_WorkEfficient(__global uint* array, __global uint* higherLevelArray, __local uint* localBlock) 
{
	// TO DO: Kernel implementation
	
	int GID = get_global_id(0);
	int LID = get_local_id(0);
	int GPID = get_group_id(0);
	
	int ldx = GPID * get_local_size(0) * 2 +  LID ;
	int rdx = GPID * get_local_size(0) * 2 + LID + get_local_size(0) ;
//	higherLevelArray[ 0 ] = 0;
	localBlock[ LID ] = array[ ldx ] ;
	localBlock[ LID + get_local_size(0) ] = array[ rdx ] ;
	barrier(CLK_LOCAL_MEM_FENCE);

	for(unsigned int stride = 1 ; stride <= get_local_size(0) ; stride=stride*2 ) 
	{
		rdx = get_local_size(0) * 2 - 1 - LID * 2 * stride  ;
		ldx = get_local_size(0) * 2 - 1 - ( 2 * LID + 1 ) * stride   ;
		
		if( ldx >= 0 )
		{
			if(stride == get_local_size(0))
				{
					array[GPID * get_local_size(0) * 2 +rdx] = localBlock[ ldx ] + localBlock[ rdx ];
					higherLevelArray[ GPID ] = localBlock[ ldx ] + localBlock[ rdx ];
					localBlock[rdx] = 0 ;
				}
			else
				{
				localBlock[rdx] = localBlock[ ldx ] + localBlock[ rdx ] ;
				}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	


	for(unsigned int stride = get_local_size(0) ; stride >= 1 ; stride=stride/2 ) 
	{
		rdx = get_local_size(0) * 2 - 1 - LID * 2 * stride  ;
		ldx = get_local_size(0) * 2 - 1 - ( 2 * LID + 1 ) * stride  ;
		
		if( ldx >= 0 )
		{
			if ( stride == 1 )
			{
				array[ GPID * get_local_size(0) * 2 + rdx - 1 ] = localBlock[ ldx ] + localBlock[ rdx ] ;
				if( ( ldx - 1 ) >= 0 )
					array[ GPID * get_local_size(0) * 2 + ldx - 1 ] = localBlock[ rdx ] ;
			}
			else
			{
				int inter = localBlock[ ldx ] + localBlock[ rdx ] ;
				localBlock[ ldx ] = localBlock[ rdx ] ;
				localBlock[ rdx ] = inter ;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_WorkEfficientAdd(__global uint* higherLevelArray, __global uint* array, __local uint* localBlock) 
{
	// TO DO: Kernel implementation (large arrays)
	// Kernel that should add the group PPS to the local PPS (Figure 14)
	
	int GID = get_global_id(0) ;
	int LID = get_local_id(0) ;
	int GPID = get_group_id(0) ;
	
	int idx = ( GPID + 1 ) * get_local_size(0) + LID;
	
	if( idx < get_global_size(0) )
	{
		localBlock[ LID ] = array[ idx ] ;
	
		array[ idx ] = localBlock[ LID ] + higherLevelArray[ GPID ] ;
	//	array[ idx ] = 0 ;
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
}