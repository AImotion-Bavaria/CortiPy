# EEG Shared Memory Acquisition

This project enables real-time acquisition of EEG data from the
Brain Products ActiCHamp amplifier using a shared memory interface.
A C++ "producer" executable streams EEG samples into shared memory,
and MATLAB "consumer" scripts read the samples efficiently without losses.

---

## How it works
1. **Producer (EEG_SharedMemoryProducer.exe)**
   - Connects to the amplifier via the Brain Products AmplifierSDK
   - Optionally performs impedance measurement until a sampling rate is set
   - Applies the requested sampling rate (FS)
   - Starts continuous acquisition in a background thread
   - Writes EEG samples into a ring buffer in shared memory

2. **Consumer (MATLAB)**
   - Connects to the same shared memory block
   - Reads EEG samples using MEX functions
   - `ReadHybridMemory` reads the requested number of samples:
       * Returns immediately with data already in the buffer
       * Waits until the remaining samples are produced
       * Ring buffer ensures no data loss if consumer is fast enough
   - Impedances can be read via `GetImpedances()`

---

## Requirements

1. **Brain Products Amplifier SDK**
   - Provide this files, in the same folder to run the code <br />
    `ActiChamp.bit`<br />
    `ActiChamp_x64.dll`<br />
    `actiCHampLib.dll`<br />
    `ActiChampX.bit`<br />
    `AmplifierSDK.dll`<br />
    `AmplifierSDK.lib` <br />
    `AmplifierSDK.h`<br />
   - For compiling the `EEG_SharedMemoryProducer.cpp` following files must be in the same folder<br />
    `AmplifierSDK.h`<br />
    `BasicExample.cpp` (only for additional example from BrainVision)<br />
    `RawDataHandlerExample.h`<br />
    `SDK.h`<br />
    `AmplifierSDK.lib`<br />
    `Amplifier_LIB.h`<br />
2. **Microsoft Visual Studio with C++ Compiler**
   - Install Visual Studio 2019/2022 (Community edition is fine)
   - During installation, select workload:
     *Desktop development with C++*
   - This installs the MSVC compiler and the **x64 Native Tools Command Prompt**

3. **MATLAB**
   - Used as consumer environment
   - Requires MEX setup for MSVC
   - Verify MATLAB sees MSVC with:<br />
      mex -setup cpp

---

## Build Instructions

### Step 1: Compile the Producer
Open the **x64 Native Tools Command Prompt for VS** and run:

cl /EHsc EEG_SharedMemoryProducer.cpp AmplifierSDK.lib /Fe:EEG_SharedMemoryProducer.exe

This builds the producer executable `EEG_SharedMemoryProducer.exe`.

### Step 2: Build the MATLAB MEX Functions
From MATLAB, compile the helper functions once:<br />

mex ReadHybridMemory.cpp<br />
mex GetImpedances.cpp<br />
mex WriteFs.cpp<br />
mex ClearSharedBuffer.cpp<br />
mex StopProducer.cpp<br />
mex WriteFs.cpp<br />
---

## Usage

### Run the Producer
Start acquisition from the producer side:

EEG_SharedMemoryProducer.exe

Console output shows amplifier connection, applied sample rate,
and status messages. It keeps running until stopped from MATLAB
or by pressing Enter in the console.

### Run the Consumer (MATLAB)
Example script `Test_sharedMem.m`:

---

## Notes
- Impedance measurement is available before fs is set.
- `ReadHybridMemory` is blocking:
  * Returns samples already in the buffer immediately
  * Waits for missing samples
  * Uses a ring buffer to handle continuous streaming
- Default buffer size = 100000 samples × 43 MAX_CHANNELS
- For long sessions, save data to disk in chunks to avoid RAM overflow
- Stop producer:
  * from MATLAB via `StopProducer()`
  * or manually in its console window (press Enter)

---

## Files
mex WriteFs.cpp
mex StopProducer.cpp
mex ReadHybridMemory.cpp
mex GetImpedances.cpp
mex ClearSharedBuffer.cpp
- `EEG_SharedMemoryProducer.cpp` — C++ shared memory producer
- `EEG_SharedMemoryProducer.exe` — compiled executable
- `ReadHybridMemory.cpp` — MEX reader with blocking behavior
- `GetImpedances.cpp` — MEX impedance reader
- `WriteFs.cpp` — MEX function to request sampling rate
- `GetAcquisitionReady.cpp` — MEX function to check readiness flag
- `ClearSharedBuffer.cpp` — MEX reset for buffer
- `StopProducer.cpp` — MEX function to signal stop
- `Test_sharedMem.m` — MATLAB example consumer script
