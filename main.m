actLvledModel = load('activityLevel_trainedModel.mat');
activity_data = load('datasets\MATLAB Mobile Data\sensorlog_20241210_141135.mat');

% Extract activity data and time vectors
accelData = activity_data.Acceleration;
angVelData = activity_data.AngularVelocity;
magFieldData = activity_data.MagneticField;
orientationData = activity_data.Orientation;
positionData = activity_data.Position;
startTime = min([accelData.Timestamp(1), angVelData.Timestamp(1), magFieldData.Timestamp(1), orientationData.Timestamp(1), positionData.Timestamp(1)]);
accelTime = seconds(accelData.Timestamp - startTime);
angVelTime = seconds(angVelData.Timestamp - startTime);
magFieldTime = seconds(magFieldData.Timestamp - startTime);
orientationTime = seconds(orientationData.Timestamp - startTime);
positionTime = seconds(positionData.Timestamp - startTime);
maxTimeLength = max([length(accelTime), length(angVelTime), length(magFieldTime), length(orientationTime), length(positionTime)]);
unifiedTime = linspace(0, max([accelTime(end), angVelTime(end), magFieldTime(end), orientationTime(end), positionTime(end)]), maxTimeLength)';

% Calculate steps
stepsTaken = calculateSteps(accelData.X, accelData.Y, accelData.Z);

% Get distance and total time
[distance, time] = calculateDistanceAndTime(positionData);

% Extract features for activity level model
accelMagnitude = sqrt(accelData.X.^2 + accelData.Y.^2 + accelData.Z.^2);
angVelMagnitude = sqrt(angVelData.X.^2 + angVelData.Y.^2 + angVelData.Z.^2);
gpsTime = seconds(positionData.Timestamp - startTime);  % Time of GPS data
speed = interp1(gpsTime, positionData.speed, accelTime, 'nearest');  % Use 'nearest' for 1 Hz data
orientationX = orientationData.X;
orientationY = orientationData.Y;
orientationZ = orientationData.Z;

% Make sure all data is the same length - pad if needed
maxLength = max([length(accelTime), length(angVelTime), length(magFieldTime), length(orientationTime)]);
accelMagnitude(end+1:maxLength) = accelMagnitude(end); 
angVelMagnitude(end+1:maxLength) = angVelMagnitude(end); 
orientationX(end+1:maxLength) = orientationX(end); 
orientationY(end+1:maxLength) = orientationY(end); 
orientationZ(end+1:maxLength) = orientationZ(end); 
speed(end+1:maxLength) = speed(end); 
featureTable = table(accelMagnitude, angVelMagnitude, speed, orientationX, orientationY, orientationZ);

% Calculate activity level using model and find how much time was spent at each level in minutes
trainedModel = load('activityLevel_trainedModel.mat'); 
[yfit, ~] = trainedModel.trainedModel.predictFcn(featureTable);
activityLevel = yfit;

% Coefficients for calories from neural network model
low_actLvl_coeff = 0.602;
moderate_actLvl_coeff = 0.273;
intense_actLvl_coeff = 0.22;
step_coeff = 0.48;

% Calculate calories burned
calories = step_coeff*stepsTaken + low_actLvl_coeff*sum(activityLevel == 1) + moderate_actLvl_coeff*sum(activityLevel == 2) + intense_actLvl_coeff*sum(activityLevel == 3);

disp('Time Elapsed: ')
disp(time);
disp('Distance Travelled: ');
disp(distance);
disp('Calories burned:');
disp(calories);

figure('WindowState', 'maximized');
plot(unifiedTime, yfit, 'r', 'LineWidth', 1.5);
xlabel('Time (seconds)');
ylabel('Activity Level');
title('Predicted Activity Level Over Time');
yticks([0 1 2 3]);
yticklabels({'No Activity', 'Light', 'Moderate', 'Intense'});
grid on;

