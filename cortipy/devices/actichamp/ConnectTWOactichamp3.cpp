#include <windows.h>
#include <iostream>
#include <vector>
#include <fstream>
//#include "ConnectionStatus.h" // Include the header for connection status
#include "mex.h"
#include "matrix.h"
#include "SDK.h" // Ensure this file is correctly included
#include "RawDataHandlerExample.h" 
#include "AmplifierSDK.h"
#include <cmath>
#include <algorithm>
#include <thread>
#include <chrono>

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

static CAmplifier amp;
bool isConnected = false; // Connection state
std::vector<std::pair<std::string, std::string>> m_vpAmpDetails;
// bool m_bIsThreadRunning;
// CRITICAL_SECTION m_CriticalSection;
static float currentSamplingRate = 0.0; 

// Thread-safe queue
std::queue<std::vector<std::vector<float>>> dataQueue;
std::mutex queueMutex;
std::condition_variable dataAvailable;
std::atomic<bool> stopThreads{false}; // Flag to stop threads



void DataAcquisitionThread(CAmplifier& amp, float samplingRate) {
    RawDataHandler rdh(amp);
    std::vector<std::vector<float>> vvfData;

    amp.StartAcquisition(RM_NORMAL);
    std::cout << "Data acquisition thread started.\n";

    while (!stopThreads.load()) {
        int nSamples = rdh.ParseRawData(amp, vvfData);
        if (nSamples > 0) {
            // Push data to queue
            std::unique_lock<std::mutex> lock(queueMutex);
            dataQueue.push(vvfData);
            dataAvailable.notify_one(); // Notify consumer
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Avoid busy waiting
        }
    }

    amp.StopAcquisition();
    std::cout << "Data acquisition thread stopped.\n";
}

void DataProcessingThread() {
    std::cout << "Data processing thread started.\n";

    while (!stopThreads.load()) {
        std::unique_lock<std::mutex> lock(queueMutex);
        dataAvailable.wait(lock, [] { return !dataQueue.empty() || stopThreads.load(); });

        while (!dataQueue.empty()) {
            auto dataChunk = dataQueue.front();
            dataQueue.pop();
            lock.unlock();

            // Process or transfer dataChunk to MATLAB
            mexPrintf("Processing data chunk of size: %d\n", dataChunk.size());
            
            // Simulate data processing (replace with actual MATLAB transfer code)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            lock.lock();
        }
    }

    std::cout << "Data processing thread stopped.\n";
}

void StartDataCollection(CAmplifier& amp, float samplingRate) {
    // Start producer and consumer threads
    std::thread acquisitionThread(DataAcquisitionThread, std::ref(amp), samplingRate);
    std::thread processingThread(DataProcessingThread);

    // Wait for user input to stop
    std::cout << "Press Enter to stop data collection...\n";
    std::cin.get();

    // Signal threads to stop
    stopThreads.store(true);
    dataAvailable.notify_all();

    // Join threads
    acquisitionThread.join();
    processingThread.join();
}

/////////////////////////////////////////


// Cleanup function to close the amplifier on exit
void CleanupOnExit() {
    mexPrintf("Running CleanupOnExit...\n");
    if (isConnected) {
        int closeRes = amp.Close();
        if (closeRes != AMP_OK) {
            mexPrintf("Error closing amplifier: %d\n", closeRes);
        } else {
            isConnected = false;
            mexPrintf("Amplifier connection closed successfully.\n");
        }
    } else {
        mexPrintf("No amplifier connection to close.\n");
    }
}


int DisplayAmpInfo(int nCount)
{
    std::string sSerialNumber;
    std::string sType;
    std::pair<std::string, std::string> pssAmpDetails;
    m_vpAmpDetails.clear();
    int nRes;
    for (int i = 0; i < nCount; i++)
    {
        // Open amplifier
        nRes = amp.Open(i);
        if (nRes != AMP_OK)
        {
            mexPrintf("\nERROR in opening amplifier: %d\n", nRes);
            return nRes;
        }
        
        // Retrieve amplifier details
        nRes = amp.GetProperty(sSerialNumber, DevicePropertyID::DPROP_CHR_SerialNumber);
        if (nRes != AMP_OK)
        {
            mexPrintf("\nERROR in GetProperty SerialNumber: %d\n", nRes);
            amp.Close();
            return nRes;
        }
        nRes = amp.GetProperty(sType, DevicePropertyID::DPROP_CHR_Type);
        if (nRes != AMP_OK)
        {
            mexPrintf("\nERROR in GetProperty Type: %d\n", nRes);
            amp.Close();
            return nRes;
        }
        amp.Close();
        
        // Store and display amplifier details
        pssAmpDetails = std::make_pair(sType, sSerialNumber);
        m_vpAmpDetails.push_back(pssAmpDetails);
        mexPrintf("\nAvailable device %d: %s %s\n", i, sType.c_str(), sSerialNumber.c_str());
    }
    
    return 0; // Indicating successful execution
}

void SetElectrodeToPassiveForAllChannels() {
    int ledControlValue = 1; // Example: 1 to enable LED, 0 to disable.
    int result = amp.SetProperty(ledControlValue, DPROP_I32_LedControl);
    
    if (result != 0) {
        printf("Failed to set LED property. Error code: %d\n", result);
    } else {
        printf("LED property set successfully.\n");
    }

    // set gain=100, needed for impedance checks with passive electodes and without active shielding. The SDK default gain=0. 
    int gain = 100;
    int nRet = amp.SetProperty(gain, DPROP_I32_ActiveShieldGain);
    if (nRet != AMP_OK)
        {
            std::cout << "\nERROR in SetProperty DPROP_I32_ActiveShieldGain:\t" << nRet;
            return;
        }


    int nAvailableModules = 0;
    // Get the number of available modules (DevicePropertyID)
    int nDeviceRet = amp.GetProperty(nAvailableModules, DPROP_I32_AvailableModules);
    if (nDeviceRet != 0) {
        mexPrintf("Failed to query available modules. Error code: %d\n", nDeviceRet);
        return;
    }

    mexPrintf("Available Modules: %d\n", nAvailableModules);

    for (int moduleID = 0; moduleID < nAvailableModules; moduleID++) {
        int nAvailableChannels = 0;
        // Query the number of available channels for the module (DevicePropertyID)
        int nModuleRet = amp.GetProperty(nAvailableChannels, DPROP_I32_AvailableChannels);

        if (nModuleRet != 0) {
            mexPrintf("Failed to get channels for module %d. Error code: %d\n", moduleID, nModuleRet);
            continue; // Skip this module if query fails
        }

        mexPrintf("Module %d has %d channels.\n", moduleID, nAvailableChannels);

        for (int channelID = 0; channelID < nAvailableChannels; channelID++) {
            // Create a variable to hold the electrode type (EL_PASSIVE)
            ElectrodeType electrodeType = EL_PASSIVE;

            // Set the electrode type to passive for the channel (ChannelPropertyID)
            int nSetRet = amp.SetProperty(electrodeType, channelID, CPROP_I32_Electrode);

            if (nSetRet != 0) {
                mexPrintf("Failed to set electrode to passive for module %d, channel %d. Error code: %d\n", moduleID, channelID, nSetRet);
            } else {
                mexPrintf("Electrode set to passive for module %d, channel %d.\n", moduleID, channelID);
            }

            // Query the electrode type for the channel (ChannelPropertyID)
            t_ElectrodeType currentElectrodeType = EL_NONE;
            int nGetRet = amp.GetProperty(currentElectrodeType, channelID, CPROP_I32_Electrode);

            if (nGetRet != 0) {
                mexPrintf("Failed to query current electrode type for module %d, channel %d. Error code: %d\n", moduleID, channelID, nGetRet);
            } else {
                mexPrintf("Current electrode type for module %d, channel %d: %d\n", moduleID, channelID, currentElectrodeType);
            }
        }
    }
}







void EnableImpedanceMeasurementForModules()
{
    
    int nAvailableModules = 0;
    amp.GetProperty(nAvailableModules, DPROP_I32_AvailableModules);

    mexPrintf("Available Modules: %d\n", nAvailableModules);

    for (int i = 0; i < nAvailableModules; i++) {
        BOOL bSupportsImpedance = FALSE;
        int nRet = amp.GetProperty(bSupportsImpedance, i, ModulePropertyID::MPROP_B32_ImpedanceMeasurement);

        if (nRet < 0) {
            mexPrintf("Failed to query impedance support for module %d. Error code: %d\n", i, nRet);
            continue;
        }

        
        if (bSupportsImpedance) {
            mexPrintf("Module %d supports impedance measurement.\n", i);

            // Create an integer variable to hold the value for enabling impedance measurement
            int enableImpedance = 1;

            // Enable impedance measurement for the module
            int nSetRet = amp.SetProperty(enableImpedance, i, ModulePropertyID::MPROP_B32_ImpedanceMeasurement);  // Pass reference to enableImpedance
            if (nSetRet < 0) {
                mexPrintf("Failed to enable impedance measurement for module %d. Error code: %d\n", i, nSetRet);
            } else {
                mexPrintf("Impedance measurement enabled for module %d.\n", i);
            }
        } else {
            mexPrintf("Module %d does NOT support impedance measurement.\n", i);
        }
    }
}







void CheckImpedances(mxArray** impedanceValues)
{
    SetElectrodeToPassiveForAllChannels();
    EnableImpedanceMeasurementForModules();
    BOOL bChannelSupportImp;
    int nAvailableChannels;
    int nImpChns = 0;
    std::vector<float> vfImpData;
        
    amp.GetProperty(nAvailableChannels, DPROP_I32_AvailableChannels);

    for (int i = 0; i < nAvailableChannels; i++)
    {
        int nRet = amp.GetProperty(bChannelSupportImp, i, CPROP_B32_ImpedanceMeasurement);
        if (bChannelSupportImp && nRet >= 0)
            nImpChns++;
    }

    if (nImpChns == 0)
    {
        mexPrintf("\nNo impedance data available. Are any electrodes attached?");
        return;
    }

    vfImpData.resize(2 * (nImpChns + 1), -1.0);
    amp.StartAcquisition(RM_IMPEDANCE);
    
    int skip = 100;
    int nRet = 1;
    while ((nRet <= AMP_OK) || (skip > 0))
    {
                nRet = amp.GetData(&vfImpData[0], vfImpData.size() * sizeof(float), vfImpData.size() * sizeof(float));
                if(nRet > 0) skip--;
                Sleep(50);
    }

    amp.StopAcquisition();

    *impedanceValues = mxCreateDoubleMatrix(nImpChns + 2, 1, mxREAL); // Extra row for GND/REF
    double* impData = mxGetPr(*impedanceValues);
    
    impData[0] = vfImpData[0]; // GND
    impData[1] = vfImpData[1]; // REF

    int row = 2; // Start from row 2 for channels
    for (int i = 2; i < vfImpData.size(); i += 2, row++)
    {
        impData[row] = (vfImpData[i + 1] < 0) ? vfImpData[i] : vfImpData[i] - vfImpData[i + 1];
    }
    mexPrintf("Available Channels: %d\n", nAvailableChannels);
    mexPrintf("Channels Supporting Impedance: %d\n", nImpChns);
    mexPrintf("Retrieved impedance data size: %d\n", vfImpData.size());


}



int SelectAmpFamily()
{
    int nFamily = 1;
    int res = SetAmplifierFamily((AmplifierFamily)nFamily);
    if (res != AMP_OK) {
        mexPrintf("\nError setting amplifier family: %d\n", res);
    }
    return res;
}

void ConnectToAmp(int nIdx)
{
    if (isConnected) {
        amp.Close(); // Close any existing connection
    }
    int res = amp.Open(nIdx);
    if (res != AMP_OK)
    {
        mexPrintf("\nError opening amplifier: %d\n", res);
        mexErrMsgIdAndTxt("ConnectToAmplifier:ErrorOpening", "Error opening amplifier.");
    }
    isConnected = true;
    mexPrintf("Successfully connected to amplifier at index %d.\n", nIdx);
}

int SearchForAmps()
{
    int nRes;
    char hwi[20];
    strcpy_s(hwi, "USB");
    std::string sHWDeviceAddress = "";

    mexPrintf("\n\tSearching for devices...\n");
    
    // Call to enumerate connected devices
    nRes = EnumerateDevices(hwi, sizeof(hwi), (const char*)sHWDeviceAddress.data(), 0);
    mexPrintf("Found %d devices.\n", nRes);
    return nRes;
}

void RecordData(float acquisitionTime, mxArray** plhs) {
    RawDataHandler rdh(amp);
    std::vector<std::vector<float>> vvfData;
    std::vector<std::vector<float>> collectedData;
    int nSamples;

    // Retrieve the current sampling rate set in the amplifier
    float samplingRate = currentSamplingRate; // Use the stored sampling rate

    // Calculate the number of samples needed based on the acquisition time and sampling rate
    int targetSampleCount = static_cast<int>(acquisitionTime * samplingRate);
    int res = amp.StartAcquisition(RM_NORMAL);
    if (res != AMP_OK) {
        mexPrintf("Failed to start acquisition: %d\n", res);
        return;
    }

    int totalSamples = 0;
    mexPrintf("Starting data acquisition at %.2f Hz...\n", samplingRate);

    // Collect data until the target sample count is reached
    while (totalSamples < targetSampleCount) {
        nSamples = rdh.ParseRawData(amp, vvfData);
        if (nSamples > 0) {
            totalSamples += nSamples;
            collectedData.insert(collectedData.end(), vvfData.begin(), vvfData.end());
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Avoid busy waiting
        }
    }

    // Stop the acquisition
    amp.StopAcquisition();
    mexPrintf("Data acquisition complete.\n");

    // Convert collectedData to MATLAB-compatible output
    int numChannels = collectedData[0].size();
    int numTotalSamples = collectedData.size();
    mxArray* matlabData = mxCreateDoubleMatrix(numTotalSamples, numChannels, mxREAL);
    double* dataPtr = mxGetPr(matlabData);

    // Copy data to MATLAB array
    for (int i = 0; i < numTotalSamples; ++i) {
        for (int j = 0; j < numChannels; ++j) {
            dataPtr[i + j * numTotalSamples] = collectedData[i][j];
        }
    }

    // Return the data to MATLAB
    mexPrintf("Data returned to MATLAB function output.\n");

    *plhs = matlabData;
}

void RecordDataAlt(float acquisitionTime) {
    RawDataHandler rdh(amp);
    std::vector<std::vector<float>> vvfData;
    std::vector<std::vector<float>> collectedData;
    int nSamples;

    // Retrieve the current sampling rate set in the amplifier
    float samplingRate = currentSamplingRate; // Use the stored sampling rate

    // Calculate the number of samples needed based on the acquisition time and sampling rate
    int targetSampleCount = static_cast<int>(acquisitionTime * samplingRate);
    int res = amp.StartAcquisition(RM_NORMAL);
    if (res != AMP_OK) {
        mexPrintf("Failed to start acquisition: %d\n", res);
        return;
    }

    int totalSamples = 0;
    mexPrintf("Starting data acquisition at %.2f Hz...\n", samplingRate);

    // Collect data until the target sample count is reached
    while (totalSamples < targetSampleCount) {
        nSamples = rdh.ParseRawData(amp, vvfData);
        if (nSamples > 0) {
            totalSamples += nSamples;
            collectedData.insert(collectedData.end(), vvfData.begin(), vvfData.end());
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Avoid busy waiting
        }
    }

    // Stop the acquisition
    amp.StopAcquisition();
    mexPrintf("Data acquisition complete.\n");

    // Convert collectedData to MATLAB-compatible output
    int numChannels = collectedData[0].size();
    int numTotalSamples = collectedData.size();
    mxArray* matlabData = mxCreateDoubleMatrix(numTotalSamples, numChannels, mxREAL);
    double* dataPtr = mxGetPr(matlabData);

    // Copy data to MATLAB array
    for (int i = 0; i < numTotalSamples; ++i) {
        for (int j = 0; j < numChannels; ++j) {
            dataPtr[i + j * numTotalSamples] = collectedData[i][j];
        }
    }

    // Save the data to the MATLAB workspace
    mexPutVariable("base", "recordedData", matlabData);
    mexPrintf("Data saved to MATLAB workspace variable 'recordedData'.\n");

    mxDestroyArray(matlabData); // Clean up
}


void ChangeSampleRate(float targetFs) {
    float fBaseRate, fSubSampleDiv;
    int nRes;

    // SDK structures for property ranges
    PropertyRange<float> prAvailableBaseRates;
    PropertyRange<float> prAvailableSubDivisors;

    // Retrieve available base sample rates
    nRes = amp.GetPropertyRange(prAvailableBaseRates, DPROP_F32_BaseSampleRate);
    if (nRes != AMP_OK) {
        mexPrintf("ERROR in GetPropertyRange BaseSampleRate: %d\n", nRes);
        return;
    }

    // Retrieve available sub-sample divisors
    nRes = amp.GetPropertyRange(prAvailableSubDivisors, DPROP_F32_SubSampleDivisor);
    if (nRes != AMP_OK) {
        mexPrintf("ERROR in GetPropertyRange SubSampleDivisor: %d\n", nRes);
        return;
    }

    // Gather all possible sampling rates
    std::vector<float> possibleRates;
    std::vector<std::pair<float, float>> rateCombinations; // Stores (base rate, sub-sample divisor)

    for (int i = 0; i < prAvailableBaseRates.ByteLength / sizeof(float); ++i) {
        float baseRate = prAvailableBaseRates.RangeArray[i];
        for (int j = 0; j < prAvailableSubDivisors.ByteLength / sizeof(float); ++j) {
            float subDivisor = prAvailableSubDivisors.RangeArray[j];
            float effectiveFs = baseRate / subDivisor;
            possibleRates.push_back(effectiveFs);
            rateCombinations.emplace_back(baseRate, subDivisor);
        }
    }

    // Find the closest possible rate to the target
    auto closestIt = std::min_element(possibleRates.begin(), possibleRates.end(),
                                      [targetFs](float a, float b) {
                                          return std::abs(a - targetFs) < std::abs(b - targetFs);
                                      });
    float closestFs = *closestIt;
    int closestIdx = std::distance(possibleRates.begin(), closestIt);

    float selectedBaseRate = rateCombinations[closestIdx].first;
    float selectedSubDiv = rateCombinations[closestIdx].second;

    // Inform the user if the exact rate isn't available
    if (std::abs(closestFs - targetFs) > 0.01) { // Allow small tolerance
        mexPrintf("Exact sampling rate %.2f Hz not available. Using closest available rate: %.2f Hz\n", targetFs, closestFs);
        mexPrintf("Available effective sampling rates are:\n");
        for (const auto& rate : possibleRates) {
            mexPrintf("\t%.2f Hz\n", rate);
        }
    }

    // Set the properties to the closest matching rates
    nRes = amp.SetProperty(selectedBaseRate, DPROP_F32_BaseSampleRate);
    if (nRes != AMP_OK) {
        mexPrintf("ERROR in SetProperty BaseSampleRate: %d\n", nRes);
        return;
    }

    nRes = amp.SetProperty(selectedSubDiv, DPROP_F32_SubSampleDivisor);
    if (nRes != AMP_OK) {
        mexPrintf("ERROR in SetProperty SubSampleDivisor: %d\n", nRes);
        return;
    }

    // Confirm the final effective sampling rate
    mexPrintf("Set Base Sample Rate: %.2f\n", selectedBaseRate);
    mexPrintf("Set Sub-Sample Divisor: %.2f\n", selectedSubDiv);
    mexPrintf("Effective Sampling Rate: %.2f Hz\n", closestFs);
    currentSamplingRate = closestFs;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /*
    mode, First entry, :        0 Connect to Amplifier (here alway 0 also for actichamp is 0)
                                1 get Impedance values
                                2 Start acquisition for data
    add Infos,Second entry:     sampling rate / Hz (if first 0)
                                0 (if first 1)
                                time (if first 2)
    */
    // Check for input arguments
//     if (nrhs != 2) {
//         mexErrMsgIdAndTxt("CInvalidInput", "Two input arguments required: command type and additional inforamtion.");
//     }

    int mode = static_cast<int>(mxGetScalar(prhs[0]));
    
//     InitializeCriticalSection(&m_CriticalSection);

    // Execute based on command type
    switch (mode) {
        case 0: { // Connect to the amplifier
            mexAtExit(CleanupOnExit);
            float addInfo = static_cast<float>(mxGetScalar(prhs[1]));
            // Select amplifier family
            int nRes = SelectAmpFamily();
            if (nRes != AMP_OK)
            {
                mexErrMsgIdAndTxt("ConnectToAmplifier:ErrorSettingFamily", "Failed to set amplifier family.");
            }
        
            // Search for amplifiers
            nRes = SearchForAmps();
            if (nRes < 1)
            {
                mexErrMsgIdAndTxt("ConnectToAmplifier:NoDevices", "No amplifiers found.");
            }
        
            // Display available amplifiers
            nRes = DisplayAmpInfo(nRes);
            if (nRes != 0)
            {
                mexErrMsgIdAndTxt("ConnectToAmplifier:ErrorDisplayingInfo", "Failed to display amplifier info.");
            }
        
            // Connect to amplifier
            ConnectToAmp(mode);
            ChangeSampleRate(addInfo);
            //CheckImpedances(&plhs[0]);
            StartDataCollection(amp, addInfo);
            break;
        }
        case 1: { // Get impedance values
             CheckImpedances(&plhs[0]);
            break;
        }
        case 2: { // Get continuous data
            // Start the recording thread
            float addInfo = static_cast<float>(mxGetScalar(prhs[1]));
//             m_bIsThreadRunning = true; // You can keep this but set it to false after data collection
            //RecordData(addInfo); // acquisitionTime 
                // Call the RecordData function
            RecordData(addInfo, &plhs[0]);
//             m_bIsThreadRunning = false; // Mark it as not running after finishing
//             DeleteCriticalSection(&m_CriticalSection);
            break;
        }
       
        case 3: {
            CleanupOnExit();
            break;
        }   
        default:
            mexErrMsgIdAndTxt("InvalidCommand", "Unknown command type.\n");
    }
  
}