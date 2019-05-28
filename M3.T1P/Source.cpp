#include "mpi.h"
#include <ostream>
#include <iostream>
#include "Timer.h"
#include <vector>

/// rows and colums length == MAX_MATRIX_LENGTH x MAX_MATRIX_LENGTH
const int MAX_MATRIX_LENGTH = 32;

MPI_Status status;

double matrix_0[MAX_MATRIX_LENGTH][MAX_MATRIX_LENGTH];
double matrix_1[MAX_MATRIX_LENGTH][MAX_MATRIX_LENGTH];
double matrix_final[MAX_MATRIX_LENGTH][MAX_MATRIX_LENGTH];

void masterThread(int& i_iter, int& j_iter, int& k_iter, int& processorID, int& processorNum
	, int& totalProcessors, int& processorDestination, int& sourceID, int& matrixRows, int& rowOffset) {
	for (i_iter = 0; i_iter < MAX_MATRIX_LENGTH; i_iter++) {
		for (j_iter = 0; j_iter < MAX_MATRIX_LENGTH; j_iter++) {
			matrix_0[i_iter][j_iter] = 1.0;
			matrix_1[i_iter][j_iter] = 1.0;
			matrix_final[i_iter][j_iter] = 0.0;
		}
	}

	/// split up rows for MPI processors
	matrixRows = MAX_MATRIX_LENGTH / totalProcessors;
	rowOffset = 0;

	/// set timer
	Timer::getInstance().addStartTime(eTimeLogType::TT_MULTIPLICATION_BEGIN, "Matric multiplication");

	/// send matrix data to workers
	for (processorDestination = 1; processorDestination <= totalProcessors; processorDestination++)
	{
		MPI_Send(&rowOffset, 1, MPI_INT, processorDestination, 1, MPI_COMM_WORLD);
		MPI_Send(&matrixRows, 1, MPI_INT, processorDestination, 1, MPI_COMM_WORLD);
		MPI_Send(&matrix_0[rowOffset][0], matrixRows*MAX_MATRIX_LENGTH, MPI_DOUBLE, processorDestination, 1, MPI_COMM_WORLD);
		MPI_Send(&matrix_1, MAX_MATRIX_LENGTH*MAX_MATRIX_LENGTH, MPI_DOUBLE, processorDestination, 1, MPI_COMM_WORLD);
		/// set new rows to be sent to next iteration
		rowOffset = rowOffset + matrixRows;
	}

	/// get all data back
	for (i_iter = 1; i_iter <= totalProcessors; i_iter++)
	{
		sourceID = i_iter;
		MPI_Recv(&rowOffset, 1, MPI_INT, sourceID, 2, MPI_COMM_WORLD, &status);
		MPI_Recv(&matrixRows, 1, MPI_INT, sourceID, 2, MPI_COMM_WORLD, &status);
		MPI_Recv(&matrix_final[rowOffset][0], matrixRows*MAX_MATRIX_LENGTH, MPI_DOUBLE, sourceID, 2, MPI_COMM_WORLD, &status);
	}

	/// finish timer for multiplication
	Timer::getInstance().addFinishTime(eTimeLogType::TT_MULTIPLICATION_BEGIN);

	/// print all results for the matrix
	// uncommment to see results
	printf("Matrix results:\n");
	for (i_iter = 0; i_iter < MAX_MATRIX_LENGTH; i_iter++) {
		for (j_iter = 0; j_iter < MAX_MATRIX_LENGTH; j_iter++)
			printf("%6.2f   ", matrix_final[i_iter][j_iter]);
		printf("\n");
	}

	/// print time taken
	Timer::getInstance().printFinalTimeSheet();

};

void workerThread(int& i_iter, int& j_iter, int& k_iter, int& processorID, int& processorNum
	, int& totalProcessors, int& processorDestination, int& sourceID, int& matrixRows, int& rowOffset) {

	sourceID = 0;
	MPI_Recv(&rowOffset, 1, MPI_INT, sourceID, 1, MPI_COMM_WORLD, &status);
	MPI_Recv(&matrixRows, 1, MPI_INT, sourceID, 1, MPI_COMM_WORLD, &status);
	MPI_Recv(&matrix_0, matrixRows*MAX_MATRIX_LENGTH, MPI_DOUBLE, sourceID, 1, MPI_COMM_WORLD, &status);
	MPI_Recv(&matrix_1, MAX_MATRIX_LENGTH*MAX_MATRIX_LENGTH, MPI_DOUBLE, sourceID, 1, MPI_COMM_WORLD, &status);

	/// per process matrix multiplication 
	for (k_iter = 0; k_iter < MAX_MATRIX_LENGTH; k_iter++)
	{
		for (i_iter = 0; i_iter < matrixRows; i_iter++)
		{
			for (j_iter = 0; j_iter < MAX_MATRIX_LENGTH; j_iter++)
				matrix_final[i_iter][k_iter] += matrix_0[i_iter][j_iter] * matrix_1[j_iter][k_iter];
		}
	}

	/// sending matrix data back to the master thread
	MPI_Send(&rowOffset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
	MPI_Send(&matrixRows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
	MPI_Send(&matrix_final, matrixRows*MAX_MATRIX_LENGTH, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);

};


int main(int argc, char **argv)
{
	int processorNum;
	int processorID;
	int totalProcessors;
	int processorDestination;
	int sourceID;
	int matrixRows;
	int rowOffset;
	int i_iter, j_iter, k_iter;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &processorID);
	MPI_Comm_size(MPI_COMM_WORLD, &processorNum);

	totalProcessors = processorNum - 1;

	/// Master Process 
	// in charge of sending and setting array data to processors
	if (processorID == 0) {
		masterThread(i_iter, j_iter, k_iter, processorID, processorNum, totalProcessors, processorDestination, sourceID, matrixRows, rowOffset);
	}

	/// All processors but master thread
	if (processorID > 0) {
		workerThread(i_iter, j_iter, k_iter, processorID, processorNum, totalProcessors, processorDestination, sourceID, matrixRows, rowOffset);
	}
	
	/// clean up MPI
	MPI_Finalize();
	return 0;
}

