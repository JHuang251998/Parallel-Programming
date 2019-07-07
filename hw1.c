#include<stdio.h>
#include<mpi.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

int compare(const void * a, const void * b);
void mergeArrays(float * arr, float * send_arr, float * double_arr, float * new_send_arr, int own_size, int recv_size);

int main(int argc, char** argv) {
	int rank, size, rc;
	int phases, neighbor;
	int global_n;			// number of elements in global list
	global_n = atoi(argv[1]);
	int local_n[size];
	int i;
	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	float arr[(global_n / size) + 1];
	float send_arr[(global_n / size) + 1];
	float double_arr[2 * ((global_n / size) + 1)];
	float new_send_arr[(global_n / size) + 1];
	
	MPI_File f_input, f_output;
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f_input);
	MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f_output);
	
	if(global_n % size == 0) {
		for(i = 0; i < size; i++) {
			local_n[i] = global_n / size;
		}
		
		MPI_File_read_at(f_input, sizeof(float) * rank * local_n[rank], &arr, local_n[rank], MPI_FLOAT, MPI_STATUS_IGNORE);
		
		//Sort with odd-even sort
		//Sort local values
		qsort(arr, local_n[rank], sizeof(float), compare);
		
		//Begin iterations
		for(phases = 0; phases < size; phases++) {
			if(phases % 2 == 0) { // Even sort
				if(rank % 2 != 0) { // Odd rank
					MPI_Send(arr, local_n[rank], MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
					MPI_Recv(arr, local_n[rank], MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				else if(rank < size-1){ // Even rank
					MPI_Recv(send_arr, local_n[rank+1], MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					mergeArrays(arr, send_arr, double_arr, new_send_arr, local_n[rank], local_n[rank+1]);
					MPI_Send(new_send_arr, local_n[rank+1], MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD);
				}
			}
			else if(phases % 2 != 0) { // Odd sort
				if(rank % 2 == 0 && rank > 0) { // Even rank
					MPI_Send(arr, local_n[rank], MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
					MPI_Recv(arr, local_n[rank], MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				else if(rank < size-1){ // Odd rank
					MPI_Recv(send_arr, local_n[rank+1], MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					mergeArrays(arr, send_arr, double_arr, new_send_arr, local_n[rank], local_n[rank+1]);
					MPI_Send(new_send_arr, local_n[rank+1], MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD);
				}
			}
		}
		
		MPI_File_write_at(f_output, sizeof(float) * rank * local_n[rank], &arr, local_n[rank], MPI_FLOAT, MPI_STATUS_IGNORE);
	}
	else {
		int diff = global_n - (global_n / size) * size;
		int position[size];
		position[0] = 0;
		while(diff > 0) {
			local_n[i] = ((global_n / size) + 1);
			diff--;
			i++;
			position[i] = position[i-1] + local_n[i-1];
		}
		while(i < size) {
			local_n[i] = global_n / size;
			if(i+1 < size) {
				position[i+1] = position[i] + local_n[i];
			}
			i++;
		}
		
		MPI_File_read_at(f_input, sizeof(float) * position[rank], &arr, local_n[rank], MPI_FLOAT, MPI_STATUS_IGNORE);
		
		qsort(arr, local_n[rank], sizeof(float), compare);
		
		for(phases = 0; phases < size; phases++) {
			if(phases % 2 == 0) { // Even sort
				if(rank % 2 != 0) { // Odd rank
					MPI_Send(arr, local_n[rank], MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
					MPI_Recv(arr, local_n[rank], MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				else if(rank < size-1){ // Even rank
					MPI_Recv(send_arr, local_n[rank+1], MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					mergeArrays(arr, send_arr, double_arr, new_send_arr, local_n[rank], local_n[rank+1]);
					MPI_Send(new_send_arr, local_n[rank+1], MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD);
				}
			}
			else if(phases % 2 != 0) { // Odd sort
				if(rank % 2 == 0 && rank > 0) { // Even rank
					MPI_Send(arr, local_n[rank], MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
					MPI_Recv(arr, local_n[rank], MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				else if(rank < size-1){ // Odd rank
					MPI_Recv(send_arr, local_n[rank+1], MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					mergeArrays(arr, send_arr, double_arr, new_send_arr, local_n[rank], local_n[rank+1]);
					MPI_Send(new_send_arr, local_n[rank+1], MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD);
				}
			}
		}
		
		MPI_File_write_at(f_output, sizeof(float) * position[rank], &arr, local_n[rank], MPI_FLOAT, MPI_STATUS_IGNORE);
	}
	
	MPI_Finalize();
}

int compare(const void * a, const void * b) {
	return ( *(float *)a - *(float *)b );
}

void mergeArrays(float * arr, float * send_arr, float * double_arr, float * new_send_arr, int own_size, int recv_size) {
	int i, j, k;
	j = 0;
	k = 0;
	
	for(i = 0; i < own_size+recv_size; i++) {
		if(j == own_size) {
			double_arr[i] = send_arr[k++];
		}
		else if(k == recv_size)  {
			double_arr[i] = arr[j++];
		}
		else if (arr[j] < send_arr[k]) {
			double_arr[i] = arr[j++];
		}
		else {
			double_arr[i] = send_arr[k++];
		}
	}
	
	for(i = 0; i < own_size; i++) {
		arr[i] = double_arr[i];
	}
	
	for(i = 0; i < recv_size; i++) {
		new_send_arr[i] = double_arr[i+own_size];
	}
}
