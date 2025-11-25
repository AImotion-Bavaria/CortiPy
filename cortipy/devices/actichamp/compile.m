% Specify the library files (complete paths)
%lib2 = 'H:\\01_Projekte\\12_EEG_Analysis_Tool\\ActiCHampSDK\\Amplifier SDK\\Bin\\x64\\AmplifierSDK.lib';

% Build the MEX file (no header files in the command)
%mex('ConnectTWOactichamp3.cpp', 'AmplifierSDK.lib');
mex(append(pwd, '\Devices\actichamp\ConnectTWOactichamp4.cpp'), append(pwd, '\Devices\actichamp\AmplifierSDK.lib'));
clear ConnectTWOactichamp4

% Connect
ConnectTWOactichamp3(0, 1000);  % Connect to amplifier at index 0, fs 45000

% Get Impedance Values:
%ImpedanceValues = ConnectTWOactichamp3(1);  % Get impedance values

% Get Continuous Data:
data = zeros(1,43);
for i=1: 2
    tic
    tempData = ConnectTWOactichamp3(2,0.5);  % Get continuous data for 1second
    data = [data ; tempData];
        % Wait until 1 second has passed (using tic/toc for precise timing)

end
plot(data(:,33))
plot(tempData(:,33))
% Disconnect:
 ConnectTWOactichamp3(3);  


% result = myMexFunction(10, 20);
% 
% nrhs would be 2 (two input arguments).
% prhs[0] would point to the 10, and prhs[1] would point to the 20.
% If you want to return a result, you could create an output in plhs[0], and nlhs would be 1.
