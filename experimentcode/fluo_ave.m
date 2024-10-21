function fluo_ave = fluo_ave(fluor,cent,allcent)
if allcent == 0
    %% this function choose the 4 directions surrounding center and averaing
    % all 4 directions. Therefore, the final data has length of the shortest
    % direction
    % x1 = cent(1);
    % y1 = cent(2);
    % fluo1 = flipud(fluor(1:x1,:,1));
    % fluo2 = fluor(x1:end,:,1);
    % fluo3 = flipud(fluor(1:y1,:,2));
    % fluo4 = fluor(y1:end,:,2);
    % l = min([x1,(1609-x1),y1,(1609-y1)]);
    % 
    % fluo_ave = (fluo1(1:l,:)+fluo2(1:l,:)+fluo3(1:l,:)+fluo4(1:l,:))/4;
    % figure
    % subplot(1,2,1)
    % hold on 
    % plot(fluor(:,:,1))
    % plot(fluor(:,:,2))
    % subplot(1,2,2)
    % hold on
    % plot(fluo1)
    % plot(fluo2)
    % plot(fluo3)
    % plot(fluo4)


    %% this function returns the largest direction
    x1 = cent(1);
    y1 = cent(2);
    fluo1 = flipud(fluor(1:x1,:,1));
    fluo2 = fluor(x1:end,:,1);
    fluo3 = flipud(fluor(1:y1,:,2));
    fluo4 = fluor(y1:end,:,2);

    l = max([x1,(1609-x1),y1,(1609-y1)]);
    if l == x1
        fluo_ave = fluo1;
    elseif l == (1609-x1)
        fluo_ave = fluo2;
    elseif l == y1
        fluo_ave = fluo3;
    else
        fluo_ave = fluo4;
    end
end


if allcent == 1
%%    this function returns the largest direction
    xall = cent(1,:,1);
    yall= cent(1,:,2);
    n = length(xall);
    
    %find largest length
    l = 0;
    for x = 1:n
        x1 = xall(x);
        y1 = yall(x);
        l_temp = max([x1,(1609-x1),y1,(1609-y1)]);
        if l_temp >l
            l = l_temp;
        end
    end
    fluo_ave = zeros(l,n);
    
    for x = 1:n
        x1 = xall(x);
        y1 = yall(x);
        fluor_temp = fluor(:,x,:);
        
        fluo1 = flipud(fluor_temp(1:x1,:,1));
        fluo2 = fluor_temp(x1:end,:,1);
        fluo3 = flipud(fluor_temp(1:y1,:,2));
        fluo4 = fluor_temp(y1:end,:,2);

        l_temp = max([x1,(1609-x1),y1,(1609-y1)]);
        if l_temp == x1
            fluo_ave(1:l_temp,x) = fluo1;
        elseif l_temp == (1609-x1)
            fluo_ave(1:l_temp,x) = fluo2;
        elseif l_temp == y1
            fluo_ave(1:l_temp,x) = fluo3;
        else
            fluo_ave(1:l_temp,x) = fluo4;
        end

    end
    
%     %% this function returns a specific direction
%     xall = cent(1,:,1);
%     yall= cent(1,:,2);
%     n = length(xall);
%     
%     %find largest length
%     l = 0;
%     for x = 1:n
%         x1 = xall(x);
%         y1 = yall(x);
%         %l_temp = max([x1,(1609-x1),y1,(1609-y1)]);
%         l_temp = 1609-y1;
%         if l_temp >l
%             l = l_temp;
%         end
%     end
%     fluo_ave = zeros(l,n);
%     
%     for x = 1:n
%         x1 = xall(x);
%         y1 = yall(x);
%         fluor_temp = fluor(:,x,:);
%         
%         fluo1 = flipud(fluor_temp(1:x1,:,1));
%         fluo2 = fluor_temp(x1:end,:,1);
%         fluo3 = flipud(fluor_temp(1:y1,:,2));
%         fluo4 = fluor_temp(y1:end,:,2);
% 
%     %    l_temp = 1609-y1;
%     l_temp = 1609-y1;
%         % length fluo1: x1; fluo2: (1609-x1); fluo3: y1; fluo4: (1609-y1)
%         fluo_ave(1:l_temp,x) = fluo4;
%     end
end
