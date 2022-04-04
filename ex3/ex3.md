# ex3

## 代价函数

![](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404162547608.png)

## gradient 梯度函数

![image-20220404162609423](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404162609423.png)

## one_vs_all

![image-20220404162645021](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404162645021.png)

## predict_all

![image-20220404162706788](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404162706788.png)



## 作业octave

### lrCostFunction

![image-20220404155528917](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404155528917.png)

```octave
h = sigmoid(X*theta)

J0 = 1/m * (-y' * log(h) - (1 - y')* log(1 - h));
Reg = lambda / 2 / m * sum(theta(2:end) .^ 2);

J = J0 + Reg
 
grad(1, :) = 1/m * (X(:,1)'* (h - y));
grad(2:end, :) = 1/m * (X(:,2:end)'* (h - y)) + lambda/m*theta(2:end, :);
```

###  oneVsAll

![image-20220404161112005](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404161112005.png)

```octave
% 自动迭代找到最优的结果
initial_theta = ones(n+1,1);
options = optimset('GradObj', 'on', 'MaxIter', 50);
% 上面两行来自hint
% num_labels是几分类的类数
for c = 1:num_labels,
  [theta] = fmincg(@(t)(lrCostFunction(t, X, (y == c), lambda)),
  initial_theta, options);
  all_theta(c,:) = theta';
end;
```

### predictOneVsAll

![image-20220404161910377](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404161910377.png)

```octave
pred = all_theta * X';
[val,ind] = max(pred, [], 1);
p = ind';
```




### predict

![image-20220404162308954](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404162308954.png)

```octave
h_layer = sigmoid(X*Theta1');
% 加一个偏置单元
h_layer = [ones(size(h_layer, 1), 1) h_layer];
o_layer = sigmoid(h_layer*Theta2');
[val, ind] = max(o_layer, [], 2);
p = ind;
```

