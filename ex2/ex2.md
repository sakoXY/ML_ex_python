# ex2

## sigmoid函数

![image-20220404142815465](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404142815465.png)

![image-20220404142829112](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404142829112.png)

## cost function(代价函数)

![image-20220404143126160](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404143126160.png)

![image-20220404143137241](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404143137241.png)

## gradient descent(梯度下降)

![image-20220404143201629](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404143201629.png)

![image-20220404143211062](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404143211062.png)



## 用训练集预测和验证

![image-20220404152808518](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404152808518.png)

## 作业octave

###  plotData

![image-20220404143524144](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404143524144.png)

```octave
% Find Indices of Positive and Negative Examples
pos = find(y == 1);
neg = find(y == 0);

% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
```



### sigmoid

![](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404144949934.png)

```octave
  g = 1 ./ (1 + exp(-z));
  % 此处不用除而用点除是因为这是矩阵之间的运算
```



### costFunction

*此处之所以有转置，是因为有矩阵乘法运算，要考虑行列是否能够相乘的情况*

![image-20220404145528180](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404145528180.png)

```octave
h = sigmoid(X * theta);

J = 1 / m * (-y' * log(h) - (1 - y)' * log(1 - h));

grad = 1 / m * X' * (h - y);

% 此处之所以有转置，是因为有矩阵乘法运算，要考虑行列是否能够相乘的情况
```



###  predict

![image-20220404151536563](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404151536563.png)

```octave
h = sigmoid(X * theta)
p = h >= 0.5;
% 因为是矩阵，所以可以直接进行比较，为真是1，为假是0
% 也可以用下面这种逐个迭算的方法
%{
for i = 1 : length(p_tem),
	if p_tem(i) >= 0.5,
		p(i) = 1;
  else,
    p(i) = 0;
	end;
end;
%}
```




### costFunctionReg

![image-20220404152146426](C:\Users\25110\AppData\Roaming\Typora\typora-user-images\image-20220404152146426.png)

```octave
h = sigmoid(X * theta);

J0 = 1 / m * (-y' * log(h) - (1 - y)' * log(1 - h));

reg = lambda / (2 * m) * theta(2:end)' * theta(2:end);

J = J0 + reg;

grad = 1 / m * X' * (h - y) + lambda / m * theta;

grad(1) = grad(1) - lambda / m * theta(1);
```

