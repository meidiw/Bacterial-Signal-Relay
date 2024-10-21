function d = timecourse_distance(data,bg)
if bg == 1
    % if background is already subtracted, use 0 as min
    col = size(data);
    d = zeros(col(2),1);
    for i = 1:col(2)
        temp = data(:,i);
        temp_nz = temp(find(temp));
        temp_max = max(temp_nz);
        temp_min = 0;
        temp_half = (temp_max - temp_min)/2;
        % include treshold: max-min >80, otherwise distance = 0
        if temp_half >40 % 0.001 for scaled 40 for not scaled
            temp_thresh = temp_half + temp_min;
            temp_above = find(temp_nz > temp_thresh);
            if ~isempty(temp_above)
                temp_start = temp_above(1);
%                 temp_below = find(temp(temp_start:end) < temp_thresh,1,'first');
                temp_below = temp_above(end);
                if ~isempty(temp_below) 
                    d(i) = temp_start + temp_below - 1;
                else
                    d(i) = col(2);
                end
            else
                d(i) = col(2);
            end
        else
            d(i) =nan;
    end
    end
end

if bg == 0
    % if background is not subtracted, use the last 20 points as min
        col = size(data);
    d = zeros(col(2),1);
    for i = 1:col(2)
        temp = data(:,i);
        temp_nz = temp(find(temp));
        temp_nz_st = sort(temp_nz, 'descend');
        temp_max = mean(temp_nz_st(1:5));% average largest 4 points
        temp_min = mean(temp_nz_st(end-20:end)); % average smallest 20 points
        temp_half = (temp_max - temp_min)/2;
        % include treshold: max-min >100, otherwise distance = 0
        if temp_half >40 % 0.001 for scaled 40 for not scaled
            temp_thresh = temp_half + temp_min;
            temp_above = find(temp_nz > temp_thresh);
            if ~isempty(temp_above)
                temp_start = temp_above(1);
                temp_below = find(temp(temp_start:end) < temp_thresh,1,'first');
                temp_below = temp_above(end);
                if ~isempty(temp_below) 
                    d(i) = temp_start + temp_below - 1;
                else
                    d(i) = col(2);
                end
            else
                d(i) = col(2);
            end
        else
            d(i) =nan;
    end
    end
end
%% 
% integral over circle
% function distance = timecourse_distance(data)
%     col = size(data);
%     distance = zeros(col(2),1);
%     for i = 1:col(2)
%         temp = data(:,i);
%         temp_nz = temp(find(temp));
%         temp_max = max(temp_nz);
%         temp_nz_st = sort(temp_nz, 'descend');
%         temp_min = min(temp_nz_st(end-50:end)); % average smallest 20 points
% %         temp_norm = (temp_nz-temp_min)./(temp_max-temp_min);
% %         temp_int = trapz(temp_norm);
%         temp_sub = temp_nz-temp_min;
%         r = [1:length(temp_nz)];
%         temp_int = trapz(r,2*pi*r.*temp_sub);
% %         distance(i) = temp_int;
%        distance(i) = temp_int/temp_max;
%     end
% end

% random threshold

