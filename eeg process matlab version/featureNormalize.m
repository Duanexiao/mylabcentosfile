function [X_norm, mu, sigma] = featureNormalize(X)
%===减去均值处理标准差

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);
% ============================================================

end
