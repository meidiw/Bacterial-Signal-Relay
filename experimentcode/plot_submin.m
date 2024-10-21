function plot_submin(data, da_min)
%% minus the same min
col = size(data);
    d = zeros(col(2),1);
    for i = 1:col(2)
        temp = data(:,i);
        temp_nz = temp(find(temp));
        temp_sub = temp_nz - da_min(i);
        plot(temp_sub)    
    end
co1= summer(col(2));
co1 = co1(end:-1:1,:);
set(gca,'colororder',co1)

end
