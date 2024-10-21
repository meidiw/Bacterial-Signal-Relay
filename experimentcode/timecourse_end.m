function data_end = timecourse_end(data,ave_num)
% find mean of basal expression
col = size(data);
data_end = zeros(1,col(2));
for i = 1:col(2)
    temp = data(:,i);
    temp_nz = temp(find(temp));
    data_end(1,i)= mean(temp_nz(end-ave_num:end));
end

