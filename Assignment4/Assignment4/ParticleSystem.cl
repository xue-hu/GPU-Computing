

float4 cross3(float4 a, float4 b){
	float4 c;
	c.x = a.y * b.z - b.y * a.z;
	c.y = a.z * b.x - b.z * a.x;
	c.z = a.x * b.y - b.x * a.y;
	c.w = 0.f;
	return c;
}

float dot3(float4 a, float4 b){
	return a.x*b.x + a.y*b.y + a.z*b.z;
}


#define EPSILON 0.001f

// This function expects two points defining a ray (x0 and x1)
// and three vertices stored in v1, v2, and v3 (the last component is not used)
// it returns true if an intersection is found and sets the isectT and isectN
// with the intersection ray parameter and the normal at the intersection point.
bool LineTriangleIntersection(	float4 x0, float4 x1,
								float4 v1, float4 v2, float4 v3,
								float *isectT, float4 *isectN){

	float4 dir = x1 - x0;
	dir.w = 0.f;

	float4 e1 = v2 - v1;
	float4 e2 = v3 - v1;
	e1.w = 0.f;
	e2.w = 0.f;

	float4 s1 = cross3(dir, e2);
	float divisor = dot3(s1, e1);
	if (divisor == 0.f)
		return false;
	float invDivisor = 1.f / divisor;

	// Compute first barycentric coordinate
	float4 d = x0 - v1;
	float b1 = dot3(d, s1) * invDivisor;
	if (b1 < -EPSILON || b1 > 1.f + EPSILON)
		return false;

	// Compute second barycentric coordinate
	float4 s2 = cross3(d, e1);
	float b2 = dot3(dir, s2) * invDivisor;
	if (b2 < -EPSILON || b1 + b2 > 1.f + EPSILON)
		return false;

	// Compute _t_ to intersection point
	float t = dot3(e2, s2) * invDivisor;
	if (t < -EPSILON || t > 1.f + EPSILON)
		return false;

	// Store the closest found intersection so far
	*isectT = t;
	*isectN = cross3(e1, e2);
	*isectN = normalize(*isectN);
	return true;

}


bool CheckCollisions(	float4 x0, float4 x1,
						__global float4 *gTriangleSoup, 
						__local float4* lTriangleCache,	// The cache should hold as many vertices as the number of threads (therefore the number of triangles is nThreads/3)
						uint nTriangles,
						float  *t,
						float4 *n){


	// ADD YOUR CODE HERE

	// Each vertex of a triangle is stored as a float4, the last component is not used.
	// gTriangleSoup contains vertices of triangles in the following layout:
	// --------------------------------------------------------------
	// | t0_v0 | t0_v1 | t0_v2 | t1_v0 | t1_v1 | t1_v2 | t2_v0 | ...
	// --------------------------------------------------------------

	// First check collisions loading the triangles from the global memory.
	// Iterate over all triangles, load the vertices and call the LineTriangleIntersection test 
	// for each triangle to find the closest intersection. 

	// Notice that each thread has to read vertices of all triangles.
	// As an optimization you should implement caching of the triangles in the local memory.
	// The cache should hold as many float4 vertices as the number of threads.
	// In other words, each thread loads (at most) *one vertex*.
	// Consequently, the number of triangles in the cache will be nThreads/4.
	// Notice that if there are many triangles (e.g. CubeMonkey.obj), not all
	// triangles can fit into the cache at once. 
	// Therefore, you have to load the triangles in chunks, until you process them all.

	// The caching implementation should roughly follow this scheme:
	//uint nProcessed = 0;  
	//while (nProcessed < nTriangles) {
	//	Load a 'k' triangles in to the cache
	//	Iterate over the triangles in the cache and test for the intersection
	//	nProcessed += k;  
	//}

	int GID = get_global_id(0) ;
	int LID = get_local_id(0) ;
	uint k = get_local_size(0) / 3 ;

	uint nProcessed = 0;
	*t = FLT_MAX ; 
	float  temp_t ;
	float4 temp_n ;

	while (nProcessed < nTriangles) {
		//Load a 'k' triangles in to the cache
		
		lTriangleCache[ LID ] = gTriangleSoup[ nProcessed * 3 + LID ] ;	
		barrier(CLK_LOCAL_MEM_FENCE) ;	
		
		//Iterate over the triangles in the cache and test for the intersection

		for(int i = 0 ; i < k ; ++i )
		{
		
			float4 v1 = lTriangleCache[ i * 3 ] ;
			float4 v2 = lTriangleCache[ i * 3 + 1 ] ;
			float4 v3 = lTriangleCache[ i * 3 + 2 ] ;
			bool interFlag = LineTriangleIntersection( x0 , x1 , v1 , v2 , v3 , &temp_t , &temp_n ) ;
			
			if( interFlag && temp_t < *t )
			{
				*t = temp_t ;
				*n = temp_n ;
			}

		}

		nProcessed += k;  
	}

	if ( *t < FLT_MAX) return true;

	return false;

}
								



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// This is the integration kernel. Implement the missing functionality
//
// Input data:
// gAlive         - Field of flag indicating whether the particle with that index is alive (!= 0) or dead (0). You will have to modify this
// gForceField    - 3D texture with the  force field
// sampler        - 3D texture sampler for the force field (see usage below)
// nParticles     - Number of input particles
// nTriangles     - Number of triangles in the scene (for collision detection)
// lTriangleCache - Local memory cache to be used during collision detection for the triangles
// gTriangleSoup  - The triangles in the scene (layout see the description of CheckCollisions())
// gPosLife       - Position (xyz) and remaining lifetime (w) of a particle
// gVelMass       - Velocity vector (xyz) and the mass (w) of a particle
// dT             - The timestep for the integration (the has to be subtracted from the remaining lifetime of each particle)
//
// Output data:
// gAlive   - Updated alive flags
// gPosLife - Updated position and lifetime
// gVelMass - Updated position and mass
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Integrate(__global uint *gAlive,
						__read_only image3d_t gForceField, 
						sampler_t sampler,
						uint nParticles,
						uint nTriangles,
						__local float4 *lTriangleCache,
						__global float4 *gTriangleSoup,
						__global float4 *gPosLife, 
						__global float4 *gVelMass,
						float dT
						)  {

	int GID = get_global_id(0) ;

	if (GID >= nParticles)
		return;
						
	float4 gAccel = (float4)(0.f, -9.81f, 0.f, 0.f);

	// Verlet Velocity Integration
	float4 x0 = gPosLife[GID];
	float4 v0 = gVelMass[GID];

	float mass = v0.w;
	float life = x0.w;

	float4 lookUp = x0;
	lookUp.w = 0.f;

	float4 F0 = read_imagef(gForceField, sampler, lookUp);
	float4 fAccel = F0 / mass ;


	// ADD YOUR CODE HERE (INSTEAD OF THE LINES BELOW) 
	// to finish the implementation of the Verlet Velocity Integration

	float4 a0 =  gAccel + fAccel ;
	float4 x1 = x0 + v0 * dT + 0.5f * a0 * dT * dT ;

	lookUp = x1;
	lookUp.w = 0.f;
	float4 F1 = read_imagef(gForceField, sampler, lookUp);
	float4 a1 =  gAccel + F1 / mass  ;
	float4 v1 = v0 + 0.5f * ( a0 + a1 ) * dT ;
	
	// Check for collisions and correct the position and velocity of the particle if it collides with a triangle
	// - Don't forget to offset the particles from the surface a little bit, otherwise they might get stuck in it.
	// - Dampen the velocity (e.g. by a factor of 0.7) to simulate dissipation of the energy.

	bool collFlag ;
	float  t ;
	float4 n ;
	collFlag = CheckCollisions( x0 , x1 ,gTriangleSoup, lTriangleCache,	// The cache should hold as many vertices as the number of threads (therefore the number of triangles is nThreads/3)
						nTriangles,&t,&n) ;

	if( collFlag )
	{
		x1 = x0 + (x1 - x0) * t + n * EPSILON;  //??pull back
		v1 = 0.7f * (v1 - 2 * ( dot3( v1 , n ) * n ) ) ;	//??
	}

	v1.w = mass;
	life -= dT ;
	x1.w = life;
	gVelMass[GID] = v1;
	gPosLife[GID] = x1;


	
	// Kill the particle if its life is < 0.0 by setting the corresponding flag in gAlive to 0.
	
	if (life < 0.f)
		gAlive[GID] = 0;
	else
		gAlive[GID] = 1;

	// Independently of the status of the particle, possibly create a new one.
	// For instance, if the particle gets too fast (or too high, or passes through some region), it is split into two...
	
	if( x1.x==0.9f && v1.x>0.9f && gAlive[GID]==1 )
	{
		gAlive[GID + nParticles] = 1;
		
		gPosLife[GID].w = gPosLife[GID].w / 2.0f ;
		gPosLife[GID + nParticles] = gPosLife[GID];

		gVelMass[GID] = gVelMass[GID] / 2.0f ;
	
		gVelMass[GID + nParticles] = gVelMass[GID] ;
		
		//gVelMass[GID + nParticles].x = gVelMass[GID].x ;
		//gVelMass[GID + nParticles].y = -gVelMass[GID].y * 2 ;
		//gVelMass[GID + nParticles].z = -gVelMass[GID].z * 2 ;
		//gVelMass[GID + nParticles].w = gVelMass[GID].w ;

	}
	else
		gAlive[GID + nParticles] = 0;

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Clear(__global float4* gPosLife, __global float4* gVelMass) {
	uint GID = get_global_id(0);
	gPosLife[GID] = 0.f;
	gVelMass[GID] = 0.f;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reorganize(	__global uint* gAlive, __global uint* gRank,	
							__global float4* gPosLifeIn,  __global float4* gVelMassIn,
							__global float4* gPosLifeOut, __global float4* gVelMassOut) {


	// Re-order the particles according to the gRank obtained from the parallel prefix sum
	// ADD YOUR CODE HERE
	int GID = get_global_id(0) ;
	
	if( gAlive[GID] )
	{
		int idx = gRank[GID] - 1 ;
		gPosLifeOut[idx] = gPosLifeIn[GID] ;
		gVelMassOut[idx] = gVelMassIn[GID] ;
	
	}
	
	
	
	
	
	
	
	

}
