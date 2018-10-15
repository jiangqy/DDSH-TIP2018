function result = callHLLabel(testL, databaseL, tB, dB)
%% normalization
if size(databaseL, 1) == 1 && size(databaseL, 2) ~= 1
    testL = testL';
    databaseL = databaseL';
end

Dhamm = hammingDist(tB, dB);
Wtrue = calcGnd(testL, databaseL);

maxHamm = max(Dhamm(:));
totalGoodPairs = sum(Wtrue(:));

% find pairs with similar codes
precision = zeros(maxHamm, 1);
recall = zeros(maxHamm, 1);
for n = 1: length(precision)
    j = (Dhamm <= ((n-1) + 00.001));
    retrievalGoodPairs = sum(Wtrue(j));
    
    retrievalPairs = sum(j(:));
    precision(n) = retrievalGoodPairs / (retrievalPairs + eps);
    recall(n) = retrievalGoodPairs / totalGoodPairs;
end

result.pre = precision;
result.rec = recall;

end