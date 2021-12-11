clc; clear all; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          hyper-parameter        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dir_num = 10;
background_dir = 45;
iter = 15;
img_path = 'da.png';
intensityLevel = 7;
line_width = 4;
line_extend = 4*line_width;
clipLimit = 0.01;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img = imread(img_path);
img_gray = rgb2gray(img);
img_gray = im2double(img_gray);
[original_h, original_w] = size(img_gray);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              test a line              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
distribution = get_distribution(line_width, 3*255/(intensityLevel-1));
line = get_line(distribution, 200);
[line_size_H, line_size_W] = size(line);
figure
imshow(uint8(line));
line = my_imrotate(line, 30, 'replicate', 'nearest');
figure
imshow(uint8(line));
line = my_imrotate_back(line, -30, line_size_H, line_size_W, 'nearest');
figure
imshow(uint8(line));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  change CLAHE into normal histequal   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%enhanced_img = histeq(img_gray);
%enhanced_img = img_gray;

%%%% CLAHE
enhanced_img = adapthisteq(img_gray,'NumTiles', [10, 10],'clipLimit',clipLimit,'Distribution','rayleigh');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

quantizedImage = reduceIntensityLevel(enhanced_img, intensityLevel);
figure
imshow(quantizedImage);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      use get_mask to get mask or load it directly      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mask_all = get_mask(img_path, dir_num, iter, background_dir);
%mask_all = load('mask_all.mat').mask_all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

LDR_mask = get_LDR_mask(quantizedImage, intensityLevel);



result = ones(original_h,original_w)*255;
result = uint8(result);

strokes = stroke.empty(0,0);

for dirs = 0 : dir_num-1
%for dirs = 5 : 5
    angle = -90+dirs*180/dir_num;
    mask = mask_all(:,:,dirs+1);

    %%%%    rotate    %%%%%%%%
    mask = my_imrotate(mask, -angle, 0, 'nearest');
    LDR_mask_rotated = my_imrotate(LDR_mask, -angle, 0, 'nearest');
    img_gray_rotated = my_imrotate(img_gray, -angle, 0, 'nearest');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [gradient_magnitude,~] = imgradient(img_gray_rotated);
    interval = 255/(intensityLevel);
    for now_intersity = 0 : intensityLevel-1
        now_LDR_mask = LDR_mask_rotated(:,:,now_intersity+1);
        Grayscale = interval*(now_intersity) + interval/2;
        [start_x, start_y, L, n] = get_line_pos_and_len(mask & now_LDR_mask, line_extend, line_width);  
        distribution = get_distribution(line_width, Grayscale);   
        for i = 1:n
            tmp_importance = (255 - Grayscale) * sum(gradient_magnitude(start_y(i) : start_y(i)+2*line_width-1 , start_x(i) : start_x(i)+ L(i) - 0),'all');
            tmp_stroke = stroke(Grayscale,angle,L(i),start_x(i),start_y(i),tmp_importance);
            strokes(end+1) = tmp_stroke;
        end
    end
end

[~, ind] = sort([strokes.importance]);
ind = flip(ind);
strokes_sorted = strokes(ind);

figure
for i = 1:size(strokes,2)
    %%%%%%%%%%% imformation of a stroke %%%%%%%%%%%%%%%%%%%
    stroke_length = strokes_sorted(i).length;
    stroke_center_intensity = strokes_sorted(i).center_intensity;
    stroke_angle = strokes_sorted(i).angle;
    stroke_start_x = strokes_sorted(i).start_x;
    stroke_start_y = strokes_sorted(i).start_y;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%% a tmp canvas %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    canvas = get_Gaussian([original_h,original_w], 3, 250);
    canvas = uint8(canvas);
    canvas = my_imrotate(canvas, -stroke_angle, 'replicate', 'nearest');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%% generate a line %%%%%%%%%%%%%%%%%%%%%%%%%%
    distribution = get_distribution(line_width, stroke_center_intensity);
    line = get_line(distribution, stroke_length);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    canvas(stroke_start_y : stroke_start_y+2*line_width-1 , stroke_start_x : stroke_start_x+ stroke_length - 1) = line;
    canvas = my_imrotate_back(canvas, stroke_angle, original_h, original_w, 'nearest');
    result = min(result, canvas);
    
    if(mod(i,50) == 1)
        imshow(result);
    end
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  functions                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function gaussian = get_Gaussian(size, var, mean)
    gaussian = (var^0.5).* randn(size) + mean;
end

function distribution = get_distribution(W, G)
    distribution = zeros(W, 2);
    center = ceil(W/2);

    for i = 1:W
        distribution(i, 1) = G + (250 - G)*(abs(i-center)/center);
        distribution(i, 2) = (250 - G)*cos(abs(i-center)*pi/W);
    end
    distribution = abs(distribution);
end

function line = get_line(distribution, length)
    W = size(distribution, 1);
    patch = get_Gaussian([2*W, length], 3, 250);
    for i = 1 : W
        patch(i, :) = get_Gaussian([1, length], distribution(i,2), distribution(i,1));
    end
    %figure
    %imshow(uint8(patch));
    line = Attenuation(patch, W);   
    line = bend(line);
    %line = im2double(line);
end

function line_out = Attenuation(line_in, W)    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % The part not written in the paper %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    order = ceil((size(line_in,2)+1)/2);
    radius = floor(W/2);
    line_out = get_Gaussian(size(line_in), 3, 250);
    for i = 1 : size(line_in,2)
        for j = 1 : W
            a = abs((1-i/order)^2)/3;
            
            b = abs((1-j/radius)^2);
            line_in(j,i)  = line_in(j,i) + (line_out(j,i)- line_in(j,i))^((a+b)^0.5)/1.5;
        end
    end
    line_out = round(line_in);
    line_out(line_out<0) = 0;
    line_out(line_out>255) = 255;
    line_out = real(line_out);
    %figure
    %imshow(uint8(line_out));
end

function line_out = bend(line_in)
    W = size(line_in,1)/2;
    length = size(line_in,2);
    
    if(length > 100)
        radius = length^2/4/W;
        bend_time = 2;
    else 
        radius = length^2/2/W;
        bend_time = 1;
    end
%%%%%%%%%%%%%%%%%%%% python version %%%%%%%%%%%%%%%%%%%%%%%%%%%
%     center = ceil(length/2);
%     line_out = line_in;
%     
%     for i = 1:length
%         offset = ((center - i)^2)/2/radius;
%         int_offset = floor(offset);
%         decimal_offset = offset-int_offset;
%         for j = 1:W
%             if j > int_offset+1
%                 line_out(j,i) = decimal_offset * line_in(j-1-int_offset, i) + (1-decimal_offset)*line_in(j-int_offset,i);
%             else 
%                 line_out(j,i) = rand*3^0.5 + 250;
%             end
%         end
%     end
%     figure
%     imshow(uint8(line_out));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % shift and interpolation
    delta = zeros(1,length);
    delta(:) = (-ceil(length/2)+1:length/2).^2/2/radius;
    p = repmat(1:2*W, length, 1)' - repmat(delta, 2*W, 1);
    
    up_y = floor(p);
    up_y(up_y < 1) = 1;
    below_y = ceil(p);
    below_y(below_y < 1) = 1;

    mask1 = p < 1; 
    mask2 =  zeros(2*W,length);
    mask2(W+1:end,:) = 1;
    
    x_axis = repmat(0:length-1, 2*W, 1);
    up = up_y + x_axis.*(2*W);
    
    below = below_y + x_axis.*(2*W);
    gaussain = get_Gaussian(size(line_in), 3, 250);
    
    line_out = line_in;
    
    % if length > 100, bend 2 times
    for i = 1:bend_time  
        if i ==1
            mask = mask1 | mask2;
        else 
            mask = mask1;
        end
        line_out = line_out(up) + (p-up_y).*(line_out(below) - line_out(up));
        line_out(mask) = gaussain(mask); 
        %figure
        %imshow(uint8(line_out));
    end
    line_out(line_out<0) = 0;
    line_out(line_out>255) = 255;
    line_out = uint8(line_out);
end

function quantizedImage = reduceIntensityLevel(originalImage, intensityLevel)
    MaxLevel = 1;
    thresh=MaxLevel/intensityLevel;
    quantizedImage = floor(originalImage./thresh);
    quantizedImage(quantizedImage == intensityLevel) = intensityLevel-1;
    quantizedImage = (quantizedImage+1)*(MaxLevel/(intensityLevel));
    %quantizedImage(quantizedImage>1) = 1;
end

function LDR_mask = get_LDR_mask(quantizedImage, intensityLevel)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  in tone.py :: LDR_single_add     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    interval = 1/(intensityLevel);
    LDR_mask = zeros(size(quantizedImage,1), size(quantizedImage,2), intensityLevel);
    
    for i = 0 : intensityLevel-1
        cumulate_mask = quantizedImage <=  interval*(i+1);
        LDR_mask(:,:,i+1) = cumulate_mask;
        %%%%%%%%% save %%%%%%%%%%%%%%%%%%%%
        filename = sprintf('LDR_mask_%d.jpg',i);
        imwrite(uint8(255*cumulate_mask),filename);
    end
    
    %save('LDR_mask.mat', 'LDR_mask');
    
end

function rotated_img = my_imrotate(img, angle, method, interpolate)
    rotated_img = padarray(img,[1 1],method,'both');
    rotated_img = imrotate(rotated_img, angle, interpolate);
end

function img = my_imrotate_back(rotated_img, angle, H, W, interpolate)
    img = imrotate(rotated_img, angle, interpolate);
    inH = size(img, 1);
    inW = size(img, 2);
    x_l = floor((inW-W)/2);
    x_r = ceil ((inW-W)/2);
    y_u = floor((inH-H)/2);
    y_d = ceil ((inH-H)/2);
    img = img(1+y_u : end-y_d, 1+x_l : end-x_r, :);
end

function [start_x, start_y, L, n] = get_line_pos_and_len(mask, extend, line_width)
    %figure    
    %imshow(mask);
    start_x = [];
    start_y = [];
    L = [];
    n = 0;
    
    [H, ~] = size(mask);
    interval = get_Gaussian([1,floor(H/line_width)], 1, line_width);
    interval = floor(interval);
    
    i = 1;
    for j = 1:size(interval,2) 
    %for i = 1:H-line_width*2
        row = mask(i,:) == 1;
        d = [true, diff(row) ~= 0, true];
        a = diff(find(d)); 
        Y = repelem(a, a);
        
        tmp_start_x = find(diff([false, row])== 1);
        tmp_L = floor((Y(tmp_start_x)-1)/2);
        tmp_n = size(tmp_start_x,2);
        %tmp_start_y = ones(1, tmp_n) * i;
       
        
        %%%%%% two horizontal line too close %%%%%
%         if tmp_n>=2
%             dist = tmp_start_x(2:end) - (tmp_start_x(1:end-1)+tmp_L(1:end-1)-1);
%             keep = dist > 1;
%             keep = [true, keep];
%             tmp_start_x = tmp_start_x(keep);
%             %tmp_start_y = tmp_start_y(keep);  
%             tmp_L = tmp_L(keep);
%             tmp_n = sum(keep);
%         end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%% extend the line %%%%%%
        f = tmp_start_x - extend;
        e = tmp_start_x + 2*tmp_L + extend;
        % boundary 
        f(f<1) = 1;
        e(e > size(row,2)) = size(row,2);
        %tmp_start_x = i + (f-1).*H;
        tmp_start_x = f;
        tmp_L = floor((e-f)/2);
        tmp_start_y = ones(1, tmp_n) * i;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        %%%%%%% append %%%%%%%%%%%%%%%%%
        start_x(end+1 : end+tmp_n) = tmp_start_x;
        start_y(end+1 : end+tmp_n) = tmp_start_y;
        L(end+1 : end+tmp_n) = tmp_L;
        n = n + tmp_n;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        i = i + interval(j);
        if(i > H-line_width*2)
            i = H-line_width*2;
        end
    end
    
end


% function img = LDR(img, n)
%     Interval = 255/n;
%     img = double(img);
%     img = uint8(img./Interval);
%     img(img == n) = n-1;
%     img = uint8((img+0.5)*Interval);
% end




















