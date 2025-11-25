#include "mex.h"
#include "SharedBuffer.h"
#include <windows.h>
#include <atomic>


static SharedBuffer* shm = nullptr;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    shm = GetSharedBuffer();  

    if (nrhs != 1 || !mxIsDouble(prhs[0]))
        mexErrMsgIdAndTxt("WriteControlBlock:InputError", "Provide a single double value for the sampling rate.");

    double fs = mxGetScalar(prhs[0]);
    shm->control.targetSamplingRate.store(static_cast<float>(fs));
}
