function [d,thd] = timecourse_distance_thd(data,thd_ratio,bg)
if bg ==0
    col = size(data);
    d = zeros(col(2),1);
    i=col(2);
    temp = data(:,i);
    temp_nz = temp(find(temp));
    temp_max = max(temp_nz);
    temp_nz_st = sort(temp_nz, 'descend');
    temp_min = min(temp_nz_st(end-20:end));
    thresh = (temp_max-temp_min)*thd_ratio;
    for i = 1:col(2)
        temp = data(:,i);
        temp_nz = temp(find(temp));
        temp_nz_st = sort(temp_nz, 'descend');
        temp_min = min(temp_nz_st(end-20:end));
        temp_nz = temp_nz-temp_min; %subtract baseline
        temp_above = find(temp_nz > thresh);
        if ~isempty(temp_above)
            temp_start = temp_above(1);
            temp_below = find(temp_nz(temp_start:end) < thresh,1,'first');
            if ~isempty(temp_below)
                d(i) = temp_start + temp_below - 1;
            else
                d(i) = NaN;
            end
        else
            d(i) = NaN;
        end
        if d(i)<100
            d(i)=NaN;
        end
    end
    thd = thresh;
    nanidx = max(find(isnan(d)));
    d(1:nanidx) = NaN;
end


if bg ==1
    col = size(data);
    d = zeros(col(2),1);
    i=col(2);
    temp = data(:,i);
    temp_nz = temp(find(temp));
    temp_max = max(temp_nz);
    temp_min = 0;
    thresh = (temp_max-temp_min)*thd_ratio;
    for i = 1:col(2)
        temp = data(:,i);
        temp_nz = temp(find(temp));
        temp_min = 0;
        temp_nz = temp_nz-temp_min; %subtract baseline
        temp_above = find(temp_nz > thresh);
        if ~isempty(temp_above)
            temp_start = temp_above(1);
            temp_below = find(temp_nz(temp_start:end) < thresh,1,'first');
            if ~isempty(temp_below)
                d(i) = temp_start + temp_below - 1;
            else
                d(i) = NaN;
            end
        else
            d(i) = NaN;
        end
        if d(i)<100
            d(i)=NaN;
        end
    end
    thd = thresh;
%     nanidx = max(find(isnan(d)));
%     d(1:nanidx) = NaN;
end
end
