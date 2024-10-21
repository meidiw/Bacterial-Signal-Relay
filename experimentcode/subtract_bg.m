function d_bg = subtract_bg(data,pt)

data_min = mean(data(end-pt:end,:));
d_bg = data-data_min;

end
