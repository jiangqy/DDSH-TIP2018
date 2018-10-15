function L01 = createOneHot(L)
k = max(L) + 1;
n = numel(L);

L01 = zeros(k, n);
index = int64((0: (n-1))' * k + L + 1);
L01(index) = 1;
L01 = L01';
end