# ConnectTWOactichamp3

ConnectTWOactichamp3 is a project aimed at facilitating the connection and data acquisition from the ActiChamp system, focusing on impedance measurements and EEG data processing.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Modifying the Code](#modifying-the-code)
- [Contributing](#contributing)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **MATLAB R2022a** or later
- **MinGW64** Compiler (C) compiler (installed vis MATLAB)
- **VS2012 redistributable** (x86 and x64)
- **VS2017 redistributable** if not used as an IDE (x86 and x64)
- **Windows SDK** Version 10.0.16299.0 on windows 7 (minimum) 

## Usage 
This files need to be in the ame folder for running the ConnectTWOactichamp3.mex64:
- ActiChamp.bit
- ActiChamp_x64.dll
- actiCHampLib.dll
- ActiChampX.bit
- AmplifierSDK.dll

% Connect

ConnectTWOactichamp3(0, 45000);  % Connect to amplifier at index 0, fs 45000

% Get Impedance Values:

ImpedanceValues = ConnectTWOactichamp3(1);  % Get impedance values (GND, REF, 32 CH)

% Get Continuous Data:

data = ConnectTWOactichamp3(2,1);  % Get continuous data for 1 second (32 CH, 8 AUX)

% Disconnect:

 ConnectTWOactichamp3(3);  % Disconnect

## Modifying the Code
For Compiling the ConnectTWOactichamp3.cpp you need the follong files in the same folder:
- AmplifierSDK.h
- BasicExample.cpp (additional example)
- RawDataHandlerExample.h
- SDK.h
- AmplifierSDK.lib
- Amplifier_LIB.h

Build the MEX file 

mex('ConnectTWOactichamp3.cpp', 'AmplifierSDK.lib');

## Contributing
The BasicExample.cpp is used as starting point. ConnectTWOactichamp3.cpp is written by Laurens Kreilinger. 