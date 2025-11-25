#include "mex.h"
#include "SharedBuffer.h"
#include <windows.h>
#include <atomic>

static SharedBuffer* shm = nullptr;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    // always remap (or at least after producer restart)
    shm = GetSharedBuffer();  // throws if mapping fails

    int nImp = shm->impSize.load(std::memory_order_acquire);
    if (nImp <= 0) {
        plhs[0] = mxCreateDoubleMatrix(0, 1, mxREAL);
        return;
    }

    plhs[0] = mxCreateDoubleMatrix(1, nImp, mxREAL);
    double* out = mxGetPr(plhs[0]);
    for (int i = 0; i < nImp; i++) {
        out[i] = shm->impedances[i];
    }
}

