function [y_ind, obj, alpha] = main(clusterings, M, k, lambda, gamma)

    [~, baseClsSegs] = getAllSegs(clusterings);
    [~,n] = size(baseClsSegs);
    
    baseClsSegs = sparse(baseClsSegs);

    eachClusters = max(clusterings);
    A = cell(1,M);
    startIdx = 1;
    for i = 1:M
        endIdx = startIdx + eachClusters(i) - 1;
        A{i} = baseClsSegs(startIdx:endIdx, :)';
        startIdx = endIdx + 1;
    end

    % k-means initialization
    Y = zeros(n, k);
    fea = baseClsSegs' * diag(sum(baseClsSegs').^(-1/2));
    now = litekmeans(fea, k);
    idx = sub2ind([n, k], (1:n)', now);
    Y(idx) = 1;

    [y_ind, obj, alpha] = solver(A, Y, lambda, gamma);

    
end