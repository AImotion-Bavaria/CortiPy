#include "mex.h"
#include "SharedBuffer.h"
#include <thread>
#include <algorithm>

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    SharedBuffer* shm = GetSharedBuffer();

    if (nrhs < 1 || !mxIsDouble(prhs[0]))
        mexErrMsgIdAndTxt("ReadHybridMemory:Input", "Provide the number of samples to read.");

    const int requested = static_cast<int>(mxGetScalar(prhs[0]));
    //if (requested <= 0 || requested > BUFFER_SIZE)
    //    mexErrMsgIdAndTxt("ReadHybridMemory:Input", "Requested samples must be > 0 and <= BUFFER_SIZE.");

    plhs[0] = mxCreateDoubleMatrix(requested, MAX_CHANNELS, mxREAL);
    double* out = mxGetPr(plhs[0]);

    int copied = 0;
    while (copied < requested) {
        const int readIdx  = shm->readIndex.load(std::memory_order_acquire);
        const int writeIdx = shm->writeIndex.load(std::memory_order_acquire);
        long available = static_cast<long>(writeIdx) - static_cast<long>(readIdx);
        if (available < 0) available = 0;

        const int toCopy = static_cast<int>(std::min<long>(available, requested - copied));
        if (toCopy > 0) {
            for (int i = 0; i < toCopy; ++i) {
                int src = (readIdx + i) % BUFFER_SIZE;
                int destRow = copied + i;
                for (int ch = 0; ch < MAX_CHANNELS; ++ch)
                    out[destRow + ch * requested] = static_cast<double>(shm->data[src][ch]);
            }
            copied += toCopy;
            shm->readIndex.store(readIdx + toCopy, std::memory_order_release);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        if (shm->control.stopRequested.load(std::memory_order_acquire) && available == 0)
            break;
    }
}
