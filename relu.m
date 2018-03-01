function X = relu(P)
    if P<=0
        X == 0;
    else
        X = P;
end