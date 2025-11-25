#include <windows.h>
#include <iostream>
#include <vector>
#include <atomic>
#include <thread>
#include <algorithm>
#include <mutex>

#include <cstddef> // offsetof
#include <iostream>
#include <iomanip>

#include "SDK.h"
#include "RawDataHandlerExample.h"
#include "SharedBuffer.h"

bool isConnected = false;




HANDLE hStopEvent = NULL;
SharedBuffer* shm = nullptr;
CAmplifier amp;
std::atomic<bool> acquisitionReady{false};
std::atomic<bool> measuringImpedances{false};


void ChangeSampleRate(float targetFs)
{
    // --- 1. Define allowed options ---
    const std::vector<float> allowedRates = {100, 200, 500, 1000, 2000, 2500,
                                             5000, 10000, 20000, 25000, 50000, 100000};

    // Map requested Fs to closest allowed option
    auto it = std::min_element(allowedRates.begin(), allowedRates.end(),
                               [targetFs](float a, float b){ return std::abs(a - targetFs) < std::abs(b - targetFs); });
    float mappedFs = *it;

    // --- 2. Get amplifier base rates and sub-divisors ---
    PropertyRange<float> prAvailableBaseRates;
    PropertyRange<float> prAvailableSubDivisors;
    int nRes = amp.GetPropertyRange(prAvailableBaseRates, DPROP_F32_BaseSampleRate);
    nRes = amp.GetPropertyRange(prAvailableSubDivisors, DPROP_F32_SubSampleDivisor);

    // --- 3. Find a base/sub combination that produces mappedFs ---
    float selectedBaseRate = 0;
    float selectedSubDiv = 0;
    bool found = false;

    for (int i = 0; i < prAvailableBaseRates.ByteLength / sizeof(float); ++i) {
        for (int j = 0; j < prAvailableSubDivisors.ByteLength / sizeof(float); ++j) {
            float eff = prAvailableBaseRates.RangeArray[i] / prAvailableSubDivisors.RangeArray[j];
            if (std::abs(eff - mappedFs) < 1e-3f) {  // allow tiny floating point error
                selectedBaseRate = prAvailableBaseRates.RangeArray[i];
                selectedSubDiv = prAvailableSubDivisors.RangeArray[j];
                found = true;
                break;
            }
        }
        if (found) break;
    }

    if (!found) {
        std::cerr << "Cannot find base/sub combination for requested Fs = " << mappedFs << " Hz\n";
        return;
    }

    // --- 4. Apply to amplifier ---
    nRes = amp.SetProperty(selectedBaseRate, DPROP_F32_BaseSampleRate);
    nRes = amp.SetProperty(selectedSubDiv, DPROP_F32_SubSampleDivisor);

    std::cout << "Requested Fs: " << targetFs << " Hz, Applied Fs: " << mappedFs << " Hz\n";
}


void DataAcquisitionThread() {
    
    RawDataHandler rdh(amp);
    std::vector<std::vector<float>> vvfData;

    // start acquisition
    int res = amp.StartAcquisition(RM_NORMAL);
    if (res != AMP_OK) {
        std::cerr << "Failed to start acquisition: " << res << "\n";
        return;
    }

    while (WaitForSingleObject(hStopEvent, 0) == WAIT_TIMEOUT &&
       !shm->control.stopRequested.load(std::memory_order_acquire)) {
        int nSamples = rdh.ParseRawData(amp, vvfData);
        if (nSamples > 0) {
            for (auto& sample : vvfData) {
                // absolute counters
                int w = shm->writeIndex.load(std::memory_order_relaxed);
                int r = shm->readIndex.load(std::memory_order_acquire);

                // check for overflow: buffer full
                if ((long)w - (long)r >= BUFFER_SIZE) {
                    // drop oldest
                    shm->readIndex.store(r + 1, std::memory_order_release);
                    shm->lostSamples.fetch_add(1, std::memory_order_relaxed);
                }

                int idx = w % BUFFER_SIZE;

                // write payload
                for (int ch = 0; ch < MAX_CHANNELS; ++ch) {
                    shm->data[idx][ch] = sample[ch];
                }

                // publish sample
                shm->writeIndex.store(w + 1, std::memory_order_release);
            }
            vvfData.clear();
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    amp.StopAcquisition();
}





int SelectAmpFamily()
{
    int nFamily = 1;
    int res = SetAmplifierFamily((AmplifierFamily)nFamily);
    if (res != AMP_OK) {
        std::cerr << "Error setting amplifier family: " << res << "\n";
    }
    return res;
}

int SearchForAmps()
{
    int nRes;
    char hwi[20];
    strcpy_s(hwi, "USB");
    std::string sHWDeviceAddress = "";

    std::cout << "Searching for devices...\n";

    nRes = EnumerateDevices(hwi, sizeof(hwi), (const char*)sHWDeviceAddress.data(), 0);
    std::cout << "Found " << nRes << " devices.\n";
    return nRes;
}

void ConnectToAmp(int nIdx)
{
    if (isConnected) amp.Close();
    int res = amp.Open(nIdx);
    if (res != AMP_OK) {
        std::cerr << "Error opening amplifier: " << res << "\n";
        exit(-1);
    }
    isConnected = true;
    std::cout << "Connected to amplifier at index " << nIdx << "\n";
}


std::vector<float> CheckImpedances()
{
    int gain = 100;
    if (amp.SetProperty(gain, DPROP_I32_ActiveShieldGain) != AMP_OK) {
        std::cout << "ERROR setting ActiveShieldGain\n";
        return {};
    }

    int nAvailableChannels;
    amp.GetProperty(nAvailableChannels, DPROP_I32_AvailableChannels);
    if (nAvailableChannels == 0) return {};

    std::vector<int> impChannels;
    for (int i = 0; i < nAvailableChannels; ++i) {
        BOOL support = FALSE;
        if (amp.GetProperty(support, i, CPROP_B32_ImpedanceMeasurement) >= 0 && support)
            impChannels.push_back(i);
    }
    if (impChannels.empty()) return {};

    // allocate buffer for GND/REF + channels (each channel gives 2 floats)
    std::vector<float> vfImpData(2 * (impChannels.size() + 1), -1.0f);
    std::vector<float> impedanceValues(impChannels.size() + 2, -1.0f);

    // start impedance measurement
    amp.StartAcquisition(RM_IMPEDANCE);

    // wait a short time for amplifier to produce data
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // get data once
    int n = amp.GetData(vfImpData.data(), vfImpData.size() * sizeof(float), vfImpData.size());
    if (n <= 0) {
        amp.StopAcquisition();
        return {};
    }

    // copy only the first value of each pair
    impedanceValues[0] = vfImpData[0]; // GND
    impedanceValues[1] = vfImpData[1]; // REF
    int row = 2;
    for (size_t i = 2; i < vfImpData.size(); i += 2, row++) {
        impedanceValues[row] = vfImpData[i];
    }

    amp.StopAcquisition();
    return impedanceValues;
}





int main() {
    // --- 0. Cleanup any leftover shared memory ---
    HANDLE hMapFile = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, "EEG_SharedMemory");
    if (hMapFile) {
        SharedBuffer* oldShm = (SharedBuffer*)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedBuffer));
        if (oldShm) UnmapViewOfFile(oldShm);
        CloseHandle(hMapFile);
        hMapFile = NULL;
    }

    // --- 1. Create and map new shared memory ---
    hMapFile = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(SharedBuffer), "EEG_SharedMemory");
    if (!hMapFile) {
        std::cerr << "Failed to create shared memory\n";
        return -1;
    }

    shm = (SharedBuffer*)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedBuffer));
    if (!shm) {
        std::cerr << "Failed to map shared memory\n";
        CloseHandle(hMapFile);
        return -1;
    }

    shm->writeIndex = 0;
    shm->control.targetSamplingRate.store(0);
    shm->control.stopRequested.store(false);
    shm->lostSamples.store(0, std::memory_order_release);





if (shm) {
    std::cout << "C++: mapped SharedBuffer at " << shm << "\n";
    std::cout << "C++: sizeof(SharedBuffer) = " << sizeof(SharedBuffer) << "\n";
    std::cout << "C++ Offsets (bytes):\n";
    std::cout << " writeIndex: " << offsetof(SharedBuffer, writeIndex) << "\n";
    std::cout << " readIndex: " << offsetof(SharedBuffer, readIndex) << "\n";
    std::cout << " data: " << offsetof(SharedBuffer, data) << "\n";
    std::cout << " lostSamples: " << offsetof(SharedBuffer, lostSamples) << "\n";
    std::cout << " control: " << offsetof(SharedBuffer, control) << "\n";
    std::cout << "   control.targetSamplingRate: "
              << (offsetof(SharedBuffer, control) + offsetof(SharedBuffer::ControlBlock, targetSamplingRate)) << "\n";
    std::cout << "   control.stopRequested: "
              << (offsetof(SharedBuffer, control) + offsetof(SharedBuffer::ControlBlock, stopRequested)) << "\n";
    std::cout << "   control.useActiveElectrodes: "
              << (offsetof(SharedBuffer, control) + offsetof(SharedBuffer::ControlBlock, useActiveElectrodes)) << "\n";
    std::cout << " impedances: " << offsetof(SharedBuffer, impedances) << "\n";
    std::cout << " impSize: " << offsetof(SharedBuffer, impSize) << "\n";
    std::cout << " acquisitionReady: " << offsetof(SharedBuffer, acquisitionReady) << "\n";

    // Print first 16 bytes as hex and first int32
    unsigned char *bytes = reinterpret_cast<unsigned char*>(shm);
    std::cout << "C++ first 16 bytes: ";
    for (int i = 0; i < 16; ++i) std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)bytes[i] << " ";
    std::cout << std::dec << "\n";
    int first_int = *reinterpret_cast<int*>(shm);
    std::cout << "C++ first int32 (little-endian): 0x" << std::hex << (first_int & 0xFFFFFFFF) << std::dec << "\n";
}



    // --- 2. Open or create stop event ---
    hStopEvent = OpenEventA(EVENT_MODIFY_STATE | SYNCHRONIZE, FALSE, "EEG_StopEvent");
    if (!hStopEvent) {
        hStopEvent = CreateEventA(NULL, TRUE, FALSE, "EEG_StopEvent");
    }
    if (!hStopEvent) {
        std::cerr << "Failed to create/open stop event\n";
        UnmapViewOfFile(shm);
        CloseHandle(hMapFile);
        return -1;
    }
    
    // --- Reset stop event so threads won't exit immediately ---
    ResetEvent(hStopEvent);

    // --- 3. Connect amplifier ---
    SelectAmpFamily();
    int nDevs = SearchForAmps();
    if (nDevs <= 0) return -1;
    ConnectToAmp(0);

// --- 4. Start impedance acquisition ---
amp.StartAcquisition(RM_IMPEDANCE);
measuringImpedances.store(true);
std::thread impedanceThread([&]() {
    std::vector<float> vfImpData(2 * (MAX_CHANNELS + 1), -1.0f); 
    std::vector<float> impedanceValues(MAX_CHANNELS + 2, -1.0f);

    while (true) {
        if (shm->control.stopRequested.load() || !measuringImpedances.load())
            break;

        int res = amp.GetData(vfImpData.data(), vfImpData.size() * sizeof(float), vfImpData.size());
        
        if (shm->control.stopRequested.load() || !measuringImpedances.load())
            break;

        if (res > 0 && shm) {
            impedanceValues[0] = vfImpData[0]; // GND
            impedanceValues[1] = vfImpData[1]; // REF

            int row = 2;
            for (size_t i = 2; i < vfImpData.size(); i += 2, row++) {
                impedanceValues[row] = (vfImpData[i] >= 0) ? vfImpData[i] : -1.0f;
            }

            int nImp = static_cast<int>(impedanceValues.size());
            for (int i = 0; i < nImp && i < MAX_CHANNELS + 2; i++)
                shm->impedances[i] = impedanceValues[i];

            shm->impSize.store(nImp, std::memory_order_release);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
});




    // --- 5. Wait for target sampling rate ---
    float fs = 0;
    while (fs <= 0 && !shm->control.stopRequested.load()) {
        fs = shm->control.targetSamplingRate.load();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (shm->control.stopRequested.load()) {
        measuringImpedances.store(false);
        SetEvent(hStopEvent);
        if (impedanceThread.joinable()) impedanceThread.join();
        amp.StopAcquisition();
        UnmapViewOfFile(shm);
        CloseHandle(hMapFile);
        return 0;
    }

    // --- 6. Stop impedance thread safely before starting normal acquisition ---
    measuringImpedances.store(false);
    SetEvent(hStopEvent);
    if (impedanceThread.joinable()) impedanceThread.join();
    ResetEvent(hStopEvent);
    amp.StopAcquisition();

    // --- 7. Apply requested sampling rate ---
    ChangeSampleRate(fs);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // --- 8. Set acquisition ready flag ---
    shm->acquisitionReady.store(true, std::memory_order_release);

    // --- 9. Start acquisition thread ---
    std::thread acquisitionThread(DataAcquisitionThread);

    // --- 10. Wait for stop request ---
    while (!shm->control.stopRequested.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    shm->control.stopRequested.store(true); // in case StopProducer didn't run
    SetEvent(hStopEvent);                   // wake any threads waiting on event
    acquisitionThread.join(); // wait until acquisition finishes

    // --- 11. Cleanup ---
    amp.Close();
    UnmapViewOfFile(shm);
    shm = nullptr;
    if (hStopEvent) {
        CloseHandle(hStopEvent);
        hStopEvent = NULL;
    }
    CloseHandle(hMapFile);

    return 0;
}
