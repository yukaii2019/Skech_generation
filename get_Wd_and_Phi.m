function [Wd, phi] = get_Wd_and_Phi(kernel_radius, kernel_size, h,w, x_norm, y_norm)
    Wd = ones(h,w,kernel_size,kernel_size);
    X_x = x_norm(kernel_radius+1 : end-kernel_radius, kernel_radius+1 : end-kernel_radius);
    X_y = y_norm(kernel_radius+1 : end-kernel_radius, kernel_radius+1 : end-kernel_radius);
    
    for i = 0 : kernel_size-1
        for j = 0 : kernel_size- 1
            Y_x = x_norm(i+1 : i+h, j+1 : j+w);
            Y_y = y_norm(i+1 : i+h, j+1 : j+w);
            Wd(:,:,i+1,j+1) = X_x.*Y_x + X_y.*Y_y;
        end
    end
    phi = sign(Wd);
    Wd = abs(Wd);
end