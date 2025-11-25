#pragma once
#include <atomic>
#include <windows.h>

#define BUFFER_SIZE 200000
#define MAX_CHANNELS 43

struct SharedBuffer {
    std::atomic<int> writeIndex;
    std::atomic<int> readIndex;
    float data[BUFFER_SIZE][MAX_CHANNELS];
    std::atomic<int> lostSamples;

    struct ControlBlock {
        // --- existing flags ---
        std::atomic<float> targetSamplingRate;    // target sampling rate to set on amplifier
        std::atomic<bool> stopRequested;           // signal to stop acquisition or processing
        std::atomic<bool> useActiveElectrodes;     // select between active or passive electrode mode

        // --- defined but not yet used ---
        std::atomic<bool> streamData{false};           // enable real-time data streaming to MATLAB
        std::atomic<bool> saveToFile{false};           // save acquired data to file
        std::atomic<bool> recalibrateAmp{false};       // trigger amplifier recalibration
        std::atomic<bool> measureImpedance{false};     // start impedance measurement routine
        std::atomic<bool> autoRangeImpedance{false};   // automatically repeat impedance until stable
        std::atomic<bool> useDCMode{false};            // switch amplifier to DC recording mode
        std::atomic<bool> enableHPFilter{false};       // enable hardware high-pass filter
        std::atomic<bool> enableLPFilter{false};       // enable hardware low-pass filter
        std::atomic<bool> enableNotchFilter{false};    // enable hardware notch filter (e.g. 50/60 Hz)
        std::atomic<bool> enableActiveShield{false};   // toggle active shielding
        std::atomic<bool> enableBiasDrive{false};      // enable bias drive circuit (driven ground)
        std::atomic<bool> useCommonReference{false};   // enable common reference mode
        std::atomic<bool> useDrivenRightLeg{false};    // enable driven right leg reference
        std::atomic<bool> enableLEDs{false};           // power on/off all LEDs on the amplifier
        std::atomic<bool> blinkLEDs{false};            // make LEDs blink for visual feedback
        std::atomic<bool> showImpedanceLEDs{false};    // use LEDs to display impedance quality
        std::atomic<bool> requestStatusUpdate{false};  // request general amplifier status info
        std::atomic<bool> requestBatteryStatus{false}; // request battery level (for portable amps)
        std::atomic<bool> requestTemperature{false};   // request internal amplifier temperature

    } control;

    float impedances[MAX_CHANNELS + 2];
    std::atomic<int> impSize;
    std::atomic<bool> acquisitionReady;
};

// ------------------------------------------------------
// Optional shared memory accessor helper
// ------------------------------------------------------
inline SharedBuffer* GetSharedBuffer() {
    HANDLE hMapFile = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, "EEG_SharedMemory");
    if (!hMapFile) {
#if defined(MX_API_VER)
        mexErrMsgIdAndTxt("SharedBuffer:OpenError", "Cannot open shared memory 'EEG_SharedMemory'.");
#else
        throw std::runtime_error("Cannot open shared memory 'EEG_SharedMemory'.");
#endif
    }

    SharedBuffer* shm = static_cast<SharedBuffer*>(
        MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedBuffer))
    );
    if (!shm) {
        CloseHandle(hMapFile);
#if defined(MX_API_VER)
        mexErrMsgIdAndTxt("SharedBuffer:MapError", "Cannot map shared memory.");
#else
        throw std::runtime_error("Cannot map shared memory.");
#endif
    }

    CloseHandle(hMapFile);
    return shm;
}
