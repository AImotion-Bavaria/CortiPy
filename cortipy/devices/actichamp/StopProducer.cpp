#include "mex.h"
#include "SharedBuffer.h"
#include <windows.h>
#include <atomic>


static SharedBuffer* shm = nullptr;

void SignalStopEvent() {
    HANDLE hStopEvent = OpenEventA(EVENT_MODIFY_STATE, FALSE, "EEG_StopEvent");
    if (hStopEvent) {
        SetEvent(hStopEvent);
        CloseHandle(hStopEvent);
    }
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    shm = GetSharedBuffer();  
    shm->control.stopRequested.store(true);  // tell producer to stop
    SignalStopEvent();                        // also notify threads immediately
}
