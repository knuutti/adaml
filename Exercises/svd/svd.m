%% SVD Practical Exercise
% ADAML 2025
% Eetu Knutars

clc; close all; clearvars;

% Task 1 (2p)

% Load data
img = load("Indian_pines.mat").indian_pines;

% Minmax normalization to [0,1]
% Normalize the image data to the range [0, 1]
% img = double(img);
img = (img - min(img(:))) / (max(img(:)) - min(img(:)));

% Create wavelengths vector based on description [nm]
wavelengths = linspace(400, 2500, size(img, 3))';

figure;

% Visualize image in RGB
% Let's use hypercube() function with the wavelengths given
subplot(1,2,1)
hcube2       = hypercube(img, wavelengths);
coloredImg2  = colorize(hcube2, "Method", "rgb", "ContrastStretching",true);
imshow(coloredImg2);
title("RGB Image")

% Visualize color map of one singular wavelength (500 nm)
subplot(1,2,2)
wl = 500;
i = find(wavelengths > wl, 1);
imshow(img(:,:,i))
title("Colormap at " + num2str(wl) + " nm")

% Visualize the spectrum of one singular pixel
figure
pixel = [3, 4];
x = img(pixel(1), pixel(2), :);
x = x(:);
plot(wavelengths, x)
xlabel("Wavelength [nm]")
axis tight; 
title("Spectrum at pixel [" + num2str(pixel) + "]")

% Reshape the image matrix to pixels X wavelength
matrixImg = reshape(img, [], size(img ,3), 1);

% Task 2 (2p)

% Normalize the data before SVD
[normMatrixImg, muX, stdX] = zscore(matrixImg);


% SVD
[U, S, V] = svd(normMatrixImg, "eco");

% Calculate variance
svals = diag(S);
variance = svals.^2 / sum(svals.^2);
cumvar = cumsum(variance);

% Plot the cumulative explained variance of singular vectors
figure; 
plot(cumvar, '-o');
ylabel("Cummulative Explained Variance");
xlabel("Singular vectors")
axis tight

% Determine reconstruction rank that explains 95% of the data
R = find(cumvar >= .95, 1);
% We get R = 27 for this particular case.

% Reconstruct image using R first singular values
reconsImg = U(:, 1:R) * S(1:R, 1:R) * V(:, 1:R)';

% Truncated sizes: 
% U*S -> 21025x27 (original: 21025x220)
% V -> 27x27 (original: 220x220)
dimReducted = 21025*27 + 27*27;
dimOriginal = prod(size(img));
disp("Reducted dimensionality: " + num2str(dimReducted))
disp("Dimensionality reduction: " + num2str(100*(1-(dimReducted/dimOriginal))) + " %")

% Compute error
rmseR = mean(rmse(normMatrixImg, reconsImg));
disp("RMSE: " + num2str(100*rmseR) + " %")

% Showcase false image of the reconstruction error
reconsImg = (reconsImg .* stdX) + muX;
figure
errorImage = abs((matrixImg - reconsImg)./matrixImg);
errorImage = reshape(errorImage, size(img, 1), size(img, 2), size(img, 3));
errorImage = sum(errorImage,3) ./ size(errorImage, 3);
imagesc(errorImage)
colorbar
title("Relative error of the reconstrucion (R="+num2str(R)+")")
axis off
axis equal