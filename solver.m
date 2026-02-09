function [y_ind, obj, alpha] = solver(A, Y, lambda, gamma)

    num_clusterings = numel(A);

    % cached variables
    alpha = zeros(num_clusterings, 1) + 1 / num_clusterings;
    B = []; XX = cell(1, num_clusterings);
    for l = 1:num_clusterings
        B = [B alpha(l)^(gamma/2)*A{l}*diag(sum(A{l}).^(-1/2))];
        XX{l} = A{l}*diag(sum(A{l}).^(-1/2));
    end

    y1 = sum(Y)';
    y_ind = vec2ind(Y')';

    for iter = 1:10
        
        [y_ind, obj_Y, y1, Y] = update_Y(B, lambda, y_ind, y1, Y);
        
        YB = cell(1,num_clusterings);
        for l = 1:num_clusterings
            YB{l} = Y' * XX{l};
        end
        for v = 1:num_clusterings
            W(v) = max(size(Y,1) - sum(diag(YB{v}*YB{v}')), eps);
            r = 1/(1-gamma);
            Wtemp(v) = (gamma*W(v))^r;
        end
        alpha = Wtemp./(sum(Wtemp,2));
        
        % objective value
        obj(iter) = sum((alpha.^gamma).*W) + lambda * sum(y1.^2);
        if iter > 1 && abs(obj(iter) - obj(iter - 1)) < 1e-5
            break;
        end
        
        B = [];
        for l = 1:num_clusterings
            B = [B alpha(l)^(gamma/2)*A{l}*diag(sum(A{l}).^(-1/2))];
        end
        
    end
end

function [y_ind, obj, y1, Y] = update_Y(B, lambda, y_ind, y1, Y)
    n = size(y_ind, 1);

    YB = Y'*B;
    for i = 1:n
        BB(i) = B(i,:)* B(i,:)';
    end  
    skk_alpha_bb = BB;

    for iter = 1:10
        yBBy = diag(YB*YB');
        obj(iter) = lambda * sum(y1 .^ 2) - sum(yBBy);
        if iter > 2 && abs(obj(iter - 1) - obj(iter)) < 1e-5
            break;
        end

        for ii = 1:n
            p = y_ind(ii);
            % avoid generating empty cluster
            if y1(p) == 1
                continue;
            end

            YBbi = YB*B(ii, :)';

            delta = lambda .* (2 * y1 + 1) - 2 * YBbi - skk_alpha_bb(ii);
            delta(p) = delta(p) - 2 * lambda + 2 * skk_alpha_bb(ii);

            [~, q] = min(delta);
            if q ~= p
                y1([p, q]) = y1([p, q]) + [-1; 1];
                y_ind(ii) = q;
                Y(ii, p) = 0;
                Y(ii, q) = 1;
                YB(p, :) = YB(p, :) - B(ii, :);
                YB(q, :) = YB(q, :) + B(ii, :);
            end
        end
    end
end
