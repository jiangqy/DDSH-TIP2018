function f = logTrick(x)
f = log(1 + exp(-abs(x))) + max(0, x);
end