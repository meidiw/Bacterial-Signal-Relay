function [avgSmallest, avgLargest] = find_minmax(matrix)
    [~, nColumns] = size(matrix);
    avgSmallest = zeros(1, nColumns);
    avgLargest = zeros(1, nColumns);
    
    for col = 1:nColumns
        columnData = matrix(:, col);
        
        % Remove zeros from the column
        nonZeroData = columnData(columnData ~= 0);
        
        % Sort the non-zero column data
        sortedColumn = sort(nonZeroData);
        
        % Average of smallest 20 non-zero numbers
        avgSmallest(col) = mean(sortedColumn(1:min(20, numel(sortedColumn))));
        
        % Average of largest 20 non-zero numbers
        avgLargest(col) = mean(sortedColumn(max(end-19, 1):end));
    end
end