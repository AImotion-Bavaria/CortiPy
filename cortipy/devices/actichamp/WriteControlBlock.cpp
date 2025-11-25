#include "mex.h"
#include <windows.h>
#include <atomic>

#define BUFFER_SIZE 100000
#define MAX_CHANNELS 43

struct SharedBuffer {
    std::atomic<int> writeIndex;
    std::atomic<int> readIndex;    // shared read pointer
    float data[BUFFER_SIZE][MAX_CHANNELS];

    struct ControlBlock {
        std::atomic<float> targetSamplingRate;
        std::atomic<bool> stopRequested;
    } control;

    float impedances[MAX_CHANNELS + 2];
    std::atomic<int> impSize;
};

static SharedBuffer* shm = nullptr;

void OpenSharedMemory() {
    if (shm) return;

    HANDLE hMapFile = OpenFileMapping(FILE_MAP_WRITE, FALSE, "EEG_SharedMemory");
    if (!hMapFile)
        mexErrMsgIdAndTxt("WriteControlBlock:OpenError", "Cannot open shared memory.");

    shm = (SharedBuffer*)MapViewOfFile(hMapFile, FILE_MAP_WRITE, 0, 0, sizeof(SharedBuffer));
    if (!shm) {
        CloseHandle(hMapFile);
        mexErrMsgIdAndTxt("WriteControlBlock:MapError", "Cannot map shared memory.");
    }
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    OpenSharedMemory();

    if (nrhs != 1 || !mxIsDouble(prhs[0]))
        mexErrMsgIdAndTxt("WriteControlBlock:InputError", "Provide a single double value for the sampling rate.");

    double fs = mxGetScalar(prhs[0]);
    shm->control.targetSamplingRate.store(static_cast<float>(fs));
}
