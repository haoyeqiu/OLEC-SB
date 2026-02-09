function sim_mat = cosine_similarity(C)
    % Calculate the norm (Euclidean length) of each row vector
    normV = sqrt(sum(C.^2, 2));
    
    % Calculate the dot product matrix
    dotProduct = C * C';
    
    % Calculate the matrix of norm products
    normProduct = normV * normV';
    
    % Calculate the cosine similarity matrix, avoiding division by zero
    sim_mat = dotProduct ./ max(normProduct, eps);
end