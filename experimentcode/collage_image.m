function image_matrix = collage_image(namepath,name,center,frame,i_range)

image_matrix = zeros(99*length(frame),1022); %99*10
%frame_f= flip(frame);
for i = 1:length(frame)
    
    [yfp,~] = imread(strcat(namepath,name{frame(i),1}));

  % yfp_ad = double(yfp).*yfp_slope + yfp_inter;
  % yfp = yfp_ad;
    x = center(1,frame(i),1);
    y = center(1,frame(i),2);
    
    image_matrix(99*i-98:99*i,:)   = yfp(y-49:y+49,x:x+1021); 
end
image_matrix = double(image_matrix); 

image_matrix_st = sort(image_matrix(:), 'descend');
image_min =i_range(1);
image_max = i_range(2);
%image_min = mean(image_matrix_st(end-50:end)); % average smallest 50 points
%image_max = mean(image_matrix_st(1:50)); % average largest 50 points
image_matrix = (image_matrix-image_min)./(image_max-image_min); % normalized  to 0 and 1 based on min and max

end