#include "mex.h"
#include "SharedBuffer.h"
#include <windows.h>
#include <atomic>
#include <cstring>   // for memset

static SharedBuffer* shm = nullptr;
static HANDLE g_hMapFile = NULL;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    shm = GetSharedBuffer();  

    // Reset indices
    shm->writeIndex.store(0, std::memory_order_release);
    shm->readIndex.store(0, std::memory_order_release);

    // Zero the buffer
    std::memset(shm->data, 0, sizeof(shm->data));

    // Reset stopRequested (optional)
    shm->control.stopRequested.store(false, std::memory_order_release);
}
