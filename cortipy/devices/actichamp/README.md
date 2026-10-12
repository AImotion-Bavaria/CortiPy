# ActiCHamp producer

This folder contains what CortiPy needs to connect to a Brain Products actiCHamp (Plus) amplifier and read impedances and EEG data from it. No BrainVision Recorder, LSL or other additional software is required.

It was developed with a 32-channel actiCHamp Plus on Windows.

## How it works

The amplifier can only be reached through the Brain Products Amplifier SDK, which is a C++ library. A small C++ program, the **producer**, therefore does the talking to the amplifier. CortiPy only starts this program and exchanges data with it through a block of shared memory.

```
 actiCHamp  --USB-->  EEG_SharedMemoryProducer.exe  -->  shared memory  -->  cortipy/devices/actichamp_device.py
                      (C++, Amplifier SDK)               "EEG_SharedMemory"  (Python, ctypes)
```

1. **Connect.** `ActiChampDevice` starts the producer. The producer opens the first amplifier it finds on USB and starts measuring impedances.
2. **Impedance.** The producer keeps writing the latest impedance values into the shared memory.
3. **Acquisition.** As soon as CortiPy sets a sampling rate, the producer stops the impedance measurement and streams the samples into a ring buffer in the shared memory. Stopping the producer releases the amplifier.

The amplifier cannot measure impedance and record EEG at the same time.

## Requirements

- Windows 10 or 11, 64 bit.
- A Brain Products actiCHamp or actiCHamp Plus, connected via USB and powered, with the Brain Products USB driver installed.
- No other program may use the amplifier at the same time, for example BrainVision Recorder.

A C++ compiler and the Brain Products Amplifier SDK are only needed if you want to change the C++ code.

## Files

| File | Purpose |
|---|---|
| `EEG_SharedMemoryProducer.exe` | The producer. Started by CortiPy. |
| `AmplifierSDK.dll`, `ActiChamp_x64.dll`, `actiCHampLib.dll` | Libraries of the Brain Products Amplifier SDK. Needed at run time next to the `.exe`. |
| `ActiChamp.bit`, `ActiChampX.bit` | Firmware that the SDK loads into the amplifier. Needed at run time next to the `.exe`. |
| `ActiChamp_x64.dll.ini` | Settings of the SDK library. Leave `Emulation=0`. |
| `EEG_SharedMemoryProducer.cpp` | Source code of the producer. |
| `SharedBuffer.h` | Layout of the shared memory: ring buffer, control block, impedances. |
| `build.bat` | Builds the producer. |

The producer writes its console output to `EEG_SharedMemoryProducer.log` in this folder.

To use a producer in a different folder, set `ActiChampPath` in the parameters. That folder must contain the `.exe`, the `.dll`, the `.bit` and the `.ini` files.

## Data format

The ring buffer holds 200000 samples with 43 columns each. On a 32-channel amplifier, columns 1-32 are the EEG channels and columns 33-40 are the AUX channels, both in µV. If CortiPy reads slower than the amplifier writes, the producer overwrites the oldest samples and counts them.

Impedances are reported in Ohm in the order GND, REF, EEG channel 1, EEG channel 2 and so on.

## Changing and compiling the producer

### Getting the Amplifier SDK

The SDK headers, the import library, the documentation and the samples are not part of this repository, because their license does not allow passing them on. You only need them to compile the producer.

1. Open https://www.brainproducts.com/downloads/more-software/ and go to **Software Development Kits**.
2. Download the **BrainVision Amplifier SDK** for actiCHamp, actiCHamp Plus and LiveAmp. You have to accept the licensing terms of Brain Products first.
3. Run the installer and keep the default folder `C:\Vision\Amplifier SDK`.

The producer in this folder was built with SDK version 1.1.0.2. If the SDK is installed somewhere else, set the variable `AMPLIFIER_SDK` to that folder before you run the build script:

```bat
set "AMPLIFIER_SDK=D:\Tools\Amplifier SDK"
```

If you install a newer SDK version, also replace the `.dll` and `.bit` files in this folder with the ones from `Bin\x64` of that SDK, so that the program and its libraries match.

### Compiling

You need Visual Studio 2019 or 2022 with the workload **Desktop development with C++**. The free Community edition is enough.

1. Make sure the producer is not running.
2. Double-click `build.bat`, or run it in a terminal:

```bat
build.bat
```

The script finds Visual Studio and the SDK by itself and replaces `EEG_SharedMemoryProducer.exe`. To keep the old program, pass a different output folder, for example `build.bat C:\temp\producer`.

### If you change SharedBuffer.h

The producer and CortiPy must use exactly the same memory layout. Otherwise CortiPy reads wrong values without any error message. After every change of `SharedBuffer.h`:

1. Compile the producer again with `build.bat`.
2. Apply the same change to the classes `ControlBlock` and `SharedBuffer` and to the constants `BUFFER_SIZE` and `MAX_CHANNELS` at the top of `cortipy/devices/actichamp_device.py`.

## Troubleshooting

| Problem | Likely cause and solution |
|---|---|
| The producer exits directly after connecting | The producer found no amplifier. Check USB cable, power and driver. Close BrainVision Recorder and other programs that use the amplifier. The log shows `Found 0 devices.` |
| A DLL is reported missing | The `.dll` and `.bit` files must be in the same folder as the `.exe`. |
| "ActiChamp shared memory layout mismatch" | `SharedBuffer.h` and `actichamp_device.py` do not match. See [If you change SharedBuffer.h](#if-you-change-sharedbufferh). |
| A previous producer is still running | End `EEG_SharedMemoryProducer.exe` in the Task Manager. |

## Disclaimer

This is an independent project. It is not affiliated with, endorsed by, or supported by Brain Products GmbH. "actiCHamp" and "Brain Products" are trademarks of their respective owners. The software is provided as is, without warranty of any kind and without liability for any damages, and is not a medical device.

## License and third-party components

The source code written for CortiPy is licensed under the Apache License 2.0, see [LICENSE](../../../LICENSE). In this folder that covers `EEG_SharedMemoryProducer.cpp`, `SharedBuffer.h` and `build.bat`.

The producer uses the **BrainVision Amplifier SDK** of Brain Products GmbH. The following files in this folder are unmodified compiled components of that SDK, version 1.1.0.2, Copyright (C) Brain Products GmbH:

- `AmplifierSDK.dll`
- `ActiChamp_x64.dll`
- `actiCHampLib.dll`
- `ActiChamp.bit`
- `ActiChampX.bit`

`EEG_SharedMemoryProducer.exe` is built with the headers, the import library and the sample file `RawDataHandlerExample.h` of the same SDK.

These components and the compiled producer are **not** covered by the Apache License. They are subject to the BrainVision Amplifier SDK Licensing Terms of Brain Products GmbH. In particular:

- They may be used for **non-commercial research purposes only**. Selling, renting, lending or leasing them, or software built on them, requires a written agreement with Brain Products.
- They must **not be used for high-risk or medically classified purposes** such as diagnosis or treatment.
- They are provided as is. All warranties and all liability for damages are excluded, as far as the law permits.
- The compiled components must not be modified, decompiled or reverse engineered.

The SDK itself is not part of this repository. Its headers, import library, documentation and samples must be downloaded from Brain Products, see [Getting the Amplifier SDK](#getting-the-amplifier-sdk). By using the ActiCHamp device in CortiPy you agree to comply with the Brain Products licensing terms.
