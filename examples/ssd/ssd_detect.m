function ssd_detect()
close all;
clc;
clear mex;
clear is_valid_handle; % to clear init_key


%% load PASCAL VOC labels
labelmap_file = 'data/VOC0712/labelmap_voc.prototxt';

%% Make sure that caffe is on the matlab path
CAFFE_ROOT = '/home/chen/Workspace/ssd';
cd(CAFFE_ROOT);
addpath([CAFFE_ROOT, '/matlab']);

% Set caffe mode
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

model_def = 'models/VGGNet/VOC0712Plus/SSD_300x300_ft/deploy.prototxt';
model_weights = 'models/VGGNet/VOC0712Plus/SSD_300x300_ft/VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.caffemodel';

net = caffe.Net(model_def,model_weights, 'test');   

% input preprocessing: 'data' is the name of the input blob == net.inputs[0]
% set net to batch size of 1
% image_resize = 300;
% net.blobs('data').reshape([image_resize,image_resize, 3, 1]);
im = imread('examples/images/fish-bike.jpg');
imshow(im);
hold on

input_data = {prepare_image(im)};

% Forward pass.
detections = net.forward(input_data);

% Parse the outputs.
det_label = detections{1}(2,:);
det_conf = detections{1}(3,:);
det_xmin = detections{1}(4,:);
det_ymin = detections{1}(5,:);
det_xmax = detections{1}(6,:);
det_ymax = detections{1}(7,:);

% Get detections with confidence higher than 0.6.
top_indices = find(det_conf >= 0.6);

top_conf = det_conf(top_indices);
top_label_indices = det_label(top_indices);
% top_labels = get_labelname(labelmap, top_label_indices)
top_xmin = det_xmin(top_indices);
top_ymin = det_ymin(top_indices);
top_xmax = det_xmax(top_indices);
top_ymax = det_ymax(top_indices);

% plot the boxes
colors = jet(21);

for i = 1:length(top_conf)
   xmin = round(top_xmin(i) * size(im, 2));
   ymin = round(top_ymin(i) * size(im, 1));
   xmax = round(top_xmax(i) * size(im, 2));
   ymax = round(top_ymax(i) * size(im, 1));
   score = top_conf(i);
   label = round(top_label_indices(i));
   display_txt = sprintf('%d: %.2f', label, score);
   coords = [xmin, ymin, xmax - xmin + 1, ymax - ymin + 1];
   color = colors(label,:);
   rectangle('Position', coords, 'EdgeColor', color, 'LineWidth', 3);
   text(double(coords(1))+2, double(coords(2)), display_txt, 'BackgroundColor', 'w');
end



% ------------------------------------------------------------------------
function crops_data = prepare_image(im)
% ------------------------------------------------------------------------
% caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
% is already in W x H x C with BGR channels
IMAGE_DIM = 300;
mean_data = repmat(reshape([123,117,104], [1,1,3]), [IMAGE_DIM, IMAGE_DIM]);

% Convert an image returned by Matlab's imread to im_data in caffe's data
% format: W x H x C with BGR channels
im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data
im_data = im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)

% oversample (4 corners, center, and their x-axis flips)
crops_data = zeros(IMAGE_DIM, IMAGE_DIM, 3, 1, 'single');
crops_data(:,:,:,1) = im_data;