function result = callHRLabel(testL, databaseL, tB, dB, param)
%% normalization
if size(databaseL, 1) == 1 && size(databaseL, 2) ~= 1
    testL = testL';
    databaseL = databaseL';
end

numTest = size(testL, 1);

%% parameters setting
topkFlag = false;
map = 0;

if isfield(param, 'topk')
    topkFlag = true;
    topk = param.topk;
    nt = numel(topk);
    topkMap = zeros(1, nt);
    topkPre = zeros(1, nt);
    topkRec = zeros(1, nt);
end

%% calculate HR task
for ii = 1: numTest
    gnd = calcGnd(testL(ii, :), databaseL);
    tsum = sum(gnd);
    if tsum == 0
        continue;
    end
    hamm = hammingDist(tB(ii, :), dB);
    [~, index] = sort(hamm);
    gnd = gnd(index);
    count = 1: tsum;
    tindex = find(gnd == 1);
    map = map + mean(count ./ tindex);
    
    if topkFlag
        for jj = 1: nt
            tgnd = gnd(1: topk(jj));
            if sum(tgnd) == 0
                continue;
            end
            tcount = 1: sum(tgnd);
            tindex = find(tgnd == 1);
            topkMap(jj) = topkMap(jj) + mean(tcount ./ tindex);
            topkPre(jj) = topkPre(jj) + sum(tgnd) / topk(jj);
            topkRec(jj) = topkRec(jj) + sum(tgnd) / tsum;
        end
    end
    
end

result.map = map / numTest;
if topkFlag
    result.topkMap = topkMap / numTest;
    result.topkPre = topkPre / numTest;
    result.topkRec = topkRec / numTest;
end

end