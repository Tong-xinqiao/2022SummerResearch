# 2022SummerResearch

Firstly we developed a Bayesian Neural Network (BNN) and tried to identify some significant mean for certain factors, and we also tried a Sparsified Neural Network (SNN) based on work of Yan Sun, Qifan Song and Faming Liang, but unfortunately these two methods cannot select consistent significant factors. The original work canbe find here:

https://github.com/sylydya/Consistent-Sparse-Deep-Learning-Theory-and-Computation.

Finally we tested LassoNet to select significant factors and then refitted DNN to evaluate the performance of the selected factors based on portfolio monthly returns and sharpe ratio. With properly tuned hyperparameters and an appropriate marking scheme for the factors, the model can select top 5 or top 10 significant factors that can explain most of the excess return. It also provide insight into the annul change of useful factors, which proved the reliability of the model. 
