function mask_all = get_mask(path, dir_num,iter, background_dir) 

    sobel_x = [-5  -4  0   4   5;
               -8 -10  0  10   8;
               -10 -20  0  20  10;
               -8 -10  0  10   8;
               -5  -4  0   4   5];
    sobel_y = sobel_x';

    kernel_radius = 2;
    kernel_size = 2*kernel_radius+1;


    img = imread(path);
    img_gray = rgb2gray(img);
    img_gray = im2double(img_gray);

    [original_h,original_w] = size(img_gray);

    if original_h>original_w
        img_gray = imresize(img_gray,[512 floor(512*original_w/original_h)]);
    else
        img_gray = imresize(img_gray,[floor(512*original_h/original_w) 512]);
    end


    [h,w] = size(img_gray);

    img_gray = padarray(img_gray, [kernel_radius kernel_radius],'replicate', 'both');

    img_normal = (img_gray - min(img_gray(:)))/(max(img_gray(:))- min(img_gray(:)));

    x_der = imfilter(img_normal, sobel_x) + 1e-12;
    y_der = imfilter(img_normal, sobel_y) + 1e-12;

    gradient_magnitude = (x_der.^2 + y_der.^2).^0.5;
    gradient_norm = gradient_magnitude/(max(gradient_magnitude(:)));

    x_norm = x_der./gradient_magnitude;
    y_norm = y_der./gradient_magnitude;

    % rotate 90
    x_norm_90 = -y_norm;
    y_norm_90 = x_norm;

    Ws = ones(h,w,kernel_size,kernel_size);
    Wm = ones(h,w,kernel_size,kernel_size);

    eta = 1;
    x = gradient_norm(kernel_radius+1 : end-kernel_radius, kernel_radius+1 : end-kernel_radius);
    for i = 0:kernel_size-1
        for j = 0:kernel_size-1
            y = gradient_norm(i+1:i+h,j+1:j+w);
            Wm(:,:,i+1,j+1) = 1/2 * (1 + tanh(eta*(y-x)));
        end
    end

    for i = 1:iter
        X = sprintf('iteration %d',i);
        disp(X);
        [Wd, phi] = get_Wd_and_Phi(kernel_radius, kernel_size, h,w, x_norm_90, y_norm_90);
        kernels = zeros(h,w,kernel_size,kernel_size);

        kernels(:,:) = phi(:,:).*Ws(:,:).*Wm(:,:).*Wd(:,:);

        x_magnitude = gradient_norm.*x_norm_90;
        y_magnitude = gradient_norm.*y_norm_90;

        x_result = zeros([h,w]);
        y_result = zeros([h,w]);


        for j = 1:h
            for k = 1:w
                kernel = reshape(kernels(j,k,:,:),[kernel_size kernel_size]);
                x_result(j,k) = sum(x_magnitude(j:j+kernel_size-1, k:k+kernel_size-1).*kernel,'all');
                y_result(j,k) = sum(y_magnitude(j:j+kernel_size-1, k:k+kernel_size-1).*kernel,'all');
            end
        end

        magnitude = (x_result.^2+ y_result.^2).^0.5;
        x_norm_new = x_result./magnitude;
        y_norm_new = y_result./magnitude;

        x_norm_90(kernel_radius+1 : end-kernel_radius, kernel_radius+1 : end-kernel_radius) = x_norm_new;
        y_norm_90(kernel_radius+1 : end-kernel_radius, kernel_radius+1 : end-kernel_radius) = y_norm_new;      
    end

    x = imresize(x_norm_90,[original_h original_w], 'nearest');
    y = imresize(y_norm_90,[original_h original_w], 'nearest');
    x(x == 0) = 1e-12;

    tan = -y./x;
    angle = atan(tan);
    angle = 180*angle./pi;

    length = 180/dir_num;
    
    if exist('background_dir','var')
        t = gradient_magnitude(kernel_radius+1 : end-kernel_radius, kernel_radius+1 : end-kernel_radius);
        t = imresize(t,[original_h original_w], 'bilinear');
        angle(t<0.4) = background_dir;
    end  
    
    mask_all = zeros(original_h, original_w, dir_num);
    
    for i = 0:dir_num-1
        filename = sprintf('mask_%d.jpg',i);
        if i == 0
            mask = (angle >= -90) .* (angle < -90 + length/2) + (angle >= 90 - length/2);
            imwrite(uint8(255*mask),filename);
            mask_all(:,:,i+1) = mask;
        else
            mask = (angle >= -90+(i-1/2)*length) .* (angle < -90+(i-1/2 + 1)*length);
            imwrite(uint8(255*mask),filename);
            mask_all(:,:,i+1) = mask;
        end
    end
    
    save('mask_all.mat', 'mask_all');
end


