/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CAssignment1.h"

#include "CSimpleArraysTask.h"
#include "CMatrixRotateTask.h"

#include <iostream>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// CAssignment1

bool CAssignment1::DoCompute()
{
	// Task 1: simple array addition.
	cout << "Running vector addition example..." << endl << endl;
	{
		size_t localWorkSize[3] = {256, 1, 1};
		CSimpleArraysTask task(1048576);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t LocalWorkSize[3] = {256, 1, 1};
		CSimpleArraysTask task(48576);
		RunComputeTask(task, LocalWorkSize);
	}

	// Task 2: matrix rotation.
	std::cout << "Running matrix rotation example..." << std::endl << std::endl;
	{
		size_t LocalWorkSize[3] = {16, 16, 1};
		CMatrixRotateTask task(2048, 1024);
		RunComputeTask(task, LocalWorkSize);
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////////
