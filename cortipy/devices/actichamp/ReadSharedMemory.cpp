#include "mex.h"
#include <windows.h>
#include <atomic>
#include <thread>

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
    HANDLE hMapFile = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, "EEG_SharedMemory");
    if (!hMapFile)
        mexErrMsgIdAndTxt("ReadHybridMemory:OpenError", "Cannot open shared memory.");
    shm = (SharedBuffer*)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedBuffer));
    CloseHandle(hMapFile);
    if (!shm)
        mexErrMsgIdAndTxt("ReadHybridMemory:MapError", "Cannot map shared memory.");
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    OpenSharedMemory();

    if (nrhs < 1 || !mxIsDouble(prhs[0]))
        mexErrMsgIdAndTxt("ReadHybridMemory:Input", "Provide the number of samples to read.");
    int nRequested = static_cast<int>(mxGetScalar(prhs[0]));
    if (nRequested <= 0 || nRequested > BUFFER_SIZE)
        mexErrMsgIdAndTxt("ReadHybridMemory:Input", "Number of samples must be >0 and <= BUFFER_SIZE.");

    int readIdx = shm->readIndex.load(std::memory_order_acquire);
    int writeIdx = shm->writeIndex.load(std::memory_order_acquire);

    // Calculate available samples
    int available = writeIdx - readIdx;
    if (available < 0) available += BUFFER_SIZE;

    int nToRead = std::min(nRequested, available);

    // Wait for remaining samples if needed
    while (nToRead < nRequested && !shm->control.stopRequested.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        writeIdx = shm->writeIndex.load(std::memory_order_acquire);
        available = writeIdx - readIdx;
        if (available < 0) available += BUFFER_SIZE;
        nToRead = std::min(nRequested, available);
    }

    if (nToRead == 0) {
        plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
        return;
    }

    // Allocate MATLAB output
    plhs[0] = mxCreateDoubleMatrix(nToRead, MAX_CHANNELS, mxREAL);
    double* out = mxGetPr(plhs[0]);

    for (int i = 0; i < nToRead; ++i) {
        int idx = (readIdx + i) % BUFFER_SIZE;
        for (int ch = 0; ch < MAX_CHANNELS; ++ch) {
            out[i + ch * nToRead] = static_cast<double>(shm->data[idx][ch]);
        }
    }

    // Advance read pointer atomically
    shm->readIndex.store((readIdx + nToRead) % BUFFER_SIZE, std::memory_order_release);
}
