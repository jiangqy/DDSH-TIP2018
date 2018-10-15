function gnd = calcGnd(testL, databaseL)
if size(databaseL, 2) ~= 1
    gnd = testL * databaseL' > 0;
else 
    Dp = repmat(testL,1,length(databaseL)) - repmat(databaseL',length(testL),1);
    gnd = Dp == 0;
end
end