#define DAMPING 0.02f

#define G_ACCEL (float4)(0.f, -9.81f, 0.f, 0.f)

#define WEIGHT_ORTHO	0.138f
#define WEIGHT_DIAG		0.097f
#define WEIGHT_ORTHO_2	0.069f
#define WEIGHT_DIAG_2	0.048f


#define ROOT_OF_2 1.4142135f
#define DOUBLE_ROOT_OF_2 2.8284271f
#define EPSILON 0.001f



///////////////////////////////////////////////////////////////////////////////
// The integration kernel
// Input data:
// width and height - the dimensions of the particle grid
// d_pos - the most recent position of the cloth particle while...
// d_prevPos - ...contains the position from the previous iteration.
// elapsedTime      - contains the elapsed time since the previous invocation of the kernel,
// prevElapsedTime  - contains the previous time step.
// simulationTime   - contains the time elapsed since the start of the simulation (useful for wind)
// All time values are given in seconds.
//
// Output data:
// d_prevPos - Input data from d_pos must be copied to this array
// d_pos     - Updated positions
///////////////////////////////////////////////////////////////////////////////
  __kernel void Integrate(unsigned int width,
						unsigned int height, 
						__global float4* d_pos,
						__global float4* d_prevPos,
						float elapsedTime,
						float prevElapsedTime,
						float simulationTime) {
							
	// Make sure the work-item does not map outside the cloth
    if(get_global_id(0) >= width || get_global_id(1) >= height)
		return;

	unsigned int particleID = get_global_id(0) + get_global_id(1) * width;
	// This is just to keep every 8th particle of the first row attached to the bar
    if(particleID > width-1 || ( particleID & ( 7 )) != 0){
	

		// ADD YOUR CODE HERE!

		// Read the positions
		float4 x0 = d_prevPos[ particleID ] ;
		float4 x1 = d_pos[ particleID ] ;
		
		float t0 = prevElapsedTime ;
		float t1 = elapsedTime ;
		float sT = simulationTime ;

		float4 x2 ;
		
		// Compute the new one position using the Verlet position integration, taking into account gravity and wind
		if (t0 != 0)
		{
			float4  v1 = ( x1 - x0 ) / t0 ;
			float4 a1 = G_ACCEL ;
			x2 = x1 + v1 * t1 + 0.5f * a1 * t1 * t1 ;
		}
		else
			x2 = x1 ;
		
		// Move the value from d_pos into d_prevPos and store the new one in d_pos
		
		d_pos[particleID] = x2;
        	d_prevPos[particleID] = x1;
		
    }
}



///////////////////////////////////////////////////////////////////////////////
// Input data:
// pos1 and pos2 - The positions of two particles
// dPostDistance  - the distance between the given particles at dPost
//
// Return data:
// correction vector for particle 1
///////////////////////////////////////////////////////////////////////////////
  float4 SatisfyConstraint(float4 pos1,
						 float4 pos2,
						 float dPostDistance){
	float4 toNeighbor = pos2 - pos1;
	return (toNeighbor - normalize(toNeighbor) * dPostDistance);
}

///////////////////////////////////////////////////////////////////////////////
// Input data:
// width and height - the dimensions of the particle grid
// dPostDistance     - the distance between two orthogonally neighboring particles at dPost
// d_posIn          - the input positions
//
// Output data:
// d_posOut - new positions must be written here
///////////////////////////////////////////////////////////////////////////////

#define TILE_X 16 
#define TILE_Y 16
#define HALOSIZE 2

__kernel __attribute__((reqd_work_group_size(TILE_X, TILE_Y, 1)))
__kernel void SatisfyConstraints(unsigned int width,
								unsigned int height, 
								float dPostDistance,
								__global float4* d_posOut,
								__global float4 const * d_posIn){
    
    if(get_global_id(0) >= width || get_global_id(1) >= height)
		return;
	
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	
	unsigned int particleID = GID.x + GID.y * width;
	
	float4 pos = d_posIn[particleID];
	
	// ADD YOUR CODE HERE!
	// Satisfy all the constraints (structural, shear, and bend).
	// You can use weights defined at the beginning of this file.
	// A ping-pong scheme is needed here, so read the values from d_posIn and store the dPosults in d_posOut
	// Hint: you should use the SatisfyConstraint helper function in the following manner:
	//SatisfyConstraint(pos, neighborpos, dPostDistance) * WEIGHT_XXX

	float4 dPos = d_posIn[ particleID ] ;
	
	if (particleID > width - 1 || (particleID & (7)) != 0)
	{
	
	// Structural constraints 
    // up
	if (GID.y > 0)
		dPos += SatisfyConstraint(pos, d_posIn[ GID.x + ( GID.y - 1 ) * width ] , dPostDistance) * WEIGHT_ORTHO;
	// down
	if (GID.y < height - 1)
		dPos += SatisfyConstraint(pos, d_posIn[ GID.x + ( GID.y + 1 ) * width ] , dPostDistance) * WEIGHT_ORTHO;
	// left
	if (GID.x > 0)
		dPos += SatisfyConstraint(pos, d_posIn[ ( GID.x - 1 ) + GID.y * width ] , dPostDistance) * WEIGHT_ORTHO;
	// right
	if (GID.x < width - 1)
		dPos += SatisfyConstraint(pos, d_posIn[ ( GID.x + 1 ) + GID.y * width ] , dPostDistance) * WEIGHT_ORTHO;


	// Shear constraints
    // four neighbours 
	if (GID.y > 0 && GID.x > 0)
		dPos += SatisfyConstraint( pos , d_posIn[ ( GID.x - 1 ) + ( GID.y - 1 ) * width ] , dPostDistance*ROOT_OF_2) * WEIGHT_DIAG;

	if (GID.y > 0 && GID.x < width - 1)
		dPos += SatisfyConstraint( pos ,d_posIn[ ( GID.x + 1 ) + ( GID.y - 1 ) * width ],dPostDistance*ROOT_OF_2) * WEIGHT_DIAG;

	if (GID.y < height - 1 && GID.x > 0)
		dPos += SatisfyConstraint( pos , d_posIn[ ( GID.x - 1 ) + ( GID.y + 1 ) * width ] ,dPostDistance*ROOT_OF_2) * WEIGHT_DIAG;

	if (GID.y < height - 1 && GID.x < width - 1)
		dPos += SatisfyConstraint( pos , d_posIn[ ( GID.x + 1 ) + ( GID.y + 1 ) * width ] , dPostDistance*ROOT_OF_2) * WEIGHT_DIAG;


	
	
	// Bend constraints
	if (GID.y > 1)
		dPos += SatisfyConstraint(pos, d_posIn[ GID.x + ( GID.y - 2 ) * width ] , dPostDistance * 2) * WEIGHT_ORTHO_2;

	if (GID.y < height - 2)
		dPos += SatisfyConstraint(pos, d_posIn[ GID.x + ( GID.y + 2 ) * width ], dPostDistance * 2) * WEIGHT_ORTHO_2;

	if (GID.x > 1)
		dPos += SatisfyConstraint(pos, d_posIn[ ( GID.x - 2 ) + GID.y * width ] , dPostDistance * 2) * WEIGHT_ORTHO_2;

	if (GID.x < width - 2)
		dPos += SatisfyConstraint(pos, d_posIn[ ( GID.x + 2 ) + GID.y * width ] , dPostDistance * 2) * WEIGHT_ORTHO_2;

	if (GID.y > 1 && GID.x > 1)
		dPos += SatisfyConstraint(pos, d_posIn[ ( GID.x - 2 ) + ( GID.y - 2 ) * width ] , dPostDistance*DOUBLE_ROOT_OF_2) * WEIGHT_DIAG_2;

	if (GID.y > 1 && GID.x < width - 2)
		dPos += SatisfyConstraint(pos, d_posIn[ ( GID.x + 2  ) + ( GID.y - 2 ) * width ] , dPostDistance*DOUBLE_ROOT_OF_2) * WEIGHT_DIAG_2;

	if (GID.y < height - 2 && GID.x > 1)
		dPos += SatisfyConstraint(pos, d_posIn[ ( GID.x - 2 ) + ( GID.y + 2 ) * width ] , dPostDistance*DOUBLE_ROOT_OF_2) * WEIGHT_DIAG_2;

	if (GID.y < height - 2 && GID.x < width - 2)
		dPos += SatisfyConstraint(pos, d_posIn[ ( GID.x + 2 ) + ( GID.y + 2 ) * width ], dPostDistance*DOUBLE_ROOT_OF_2) * WEIGHT_DIAG_2;
	
	
	}
	
	d_posOut[ particleID ] = dPos ;

}


float dot3(float4 a, float4 b) 
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}


///////////////////////////////////////////////////////////////////////////////
// Input data:
// width and height - the dimensions of the particle grid
// d_pos            - the input positions
// spherePos        - The position of the sphere (xyz)
// sphereRad        - The radius of the sphere
//
// Output data:
// d_pos            - The updated positions
///////////////////////////////////////////////////////////////////////////////
__kernel void CheckCollisions(unsigned int width,
								unsigned int height, 
								__global float4* d_pos,
								float4 spherePos,
								float sphereRad){
								

	// ADD YOUR CODE HERE!
	// Find whether the particle is inside the sphere.
	// If so, push it outside.
	
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	
	if(GID.x >= width || GID.y >= height)
		return;
	
	unsigned int particleID = GID.x + GID.y * width;
	
	float4 pos = d_pos[ particleID ] ;
	float4 vec = pos - spherePos ;
	float dis = dot3( vec , vec ) ;
	
	if( dis <= ( sphereRad * sphereRad ) )
	{
		pos = spherePos + normalize( vec ) * ( sphereRad + EPSILON ) ;
		d_pos[ particleID ] = pos ;
	}

	
}

///////////////////////////////////////////////////////////////////////////////
// There is no need to change this function!
///////////////////////////////////////////////////////////////////////////////
float4 CalcTriangleNormal( float4 p1, float4 p2, float4 p3) {
    float4 v1 = p2-p1;
    float4 v2 = p3-p1;

    return cross( v1, v2);
}

///////////////////////////////////////////////////////////////////////////////
// There is no need to change this kernel!
///////////////////////////////////////////////////////////////////////////////
__kernel void ComputeNormals(unsigned int width,
								unsigned int height, 
								__global float4* d_pos,
								__global float4* d_normal){
								
    int particleID = get_global_id(0) + get_global_id(1) * width;
    float4 normal = (float4)( 0.0f, 0.0f, 0.0f, 0.0f);
    
    int minX, maxX, minY, maxY, cntX, cntY;
    minX = max( (int)(0), (int)(get_global_id(0)-1));
    maxX = min( (int)(width-1), (int)(get_global_id(0)+1));
    minY = max( (int)(0), (int)(get_global_id(1)-1));
    maxY = min( (int)(height-1), (int)(get_global_id(1)+1));
    
    for( cntX = minX; cntX < maxX; ++cntX) {
        for( cntY = minY; cntY < maxY; ++cntY) {
            normal += normalize( CalcTriangleNormal(
                d_pos[(cntX+1)+width*(cntY)],
                d_pos[(cntX)+width*(cntY)],
                d_pos[(cntX)+width*(cntY+1)]));
            normal += normalize( CalcTriangleNormal(
                d_pos[(cntX+1)+width*(cntY+1)],
                d_pos[(cntX+1)+width*(cntY)],
                d_pos[(cntX)+width*(cntY+1)]));
        }
    }
    d_normal[particleID] = normalize( normal);
}
