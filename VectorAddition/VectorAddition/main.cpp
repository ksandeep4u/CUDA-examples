#include "cpuProgram.h"
#include "gpuProgram.cuh"

using namespace std;

int main() {
	int N = 1 << 20; //number of integers to add

	// run vector addition on CPU
	cpuProgram(N);

	// run vector addition on GPU
	gpuProgram(N);

	return 0;
}
