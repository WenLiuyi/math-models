# 时间序列
## 1. 概念
时间序列：$$
X_1,X_2,...,X_T
$$
### 1.1 自协方差
$$
Cov[X_t,X_{t-h}]=\frac{1}{n-1}\sum_{t=1}^{T}(X_t-\mu)(X_{t-h}-\mu)
$$
这里的h被称为滞后。滞后的X是前一个X值偏移了h位置。
### 1.2 自相关
$$
Cor[X_t,X_{t-h}]=\frac{Cov[X_t,X_{t-h}]}{\sqrt{V[X_t]V[X_{t-h}]}}
$$
其中，$V[X_t]=\sum_{t=1}^{T}(X_t-\mu)^{2}$

### 1.3 弱平稳（二阶平稳）
$$
Cov[X_{t+\tau},X_t]=\gamma_{|t|}
$$
其中µ是常数，且 𝛾ₜ 不依赖于𝛕。这些公式表明，随着时间的推移，均值和方差是稳定的，协方差取决于时滞。

### 1.4 严格平稳性(强平稳性)

## 2. 时间序列过程
### 2.1 白噪声
$$
E[X_t]=0
$$
$$
Var[X_t]=\sigma^{2}
$$
$$
Cov[X_{t_1},X_{t_2}]=0, 对任意t_1,t_2都成立
$$
白噪声的均值为零，其方差在时间步长上是相同的。它具有零协方差，这意味着时间序列与其滞后版本是不相关的。所以自相关也是零。一般用于时间序列回归分析中残差项满足的假设。

## 2.2 自回归（AR）
AR(1)如下：$$
Y_t=\phi_0+\phi_1 Y_{t-1}+U_t, t=1,...,T
$$
可得：$$
Y_t=\phi_0+\phi_1(\phi_0+\phi_1 Y_{t-2}+U_{t-1})+U_t=\sum_{i-1}^{t}\phi_{1}^{t-i}\phi_0+\phi_{1}^{t}Y_0+\sum_{i=1}^{t}\phi_{1}^{t-i}U_i
$$
如果| 𝜙₁ | < 1，则过去值的影响随着步骤的增加而变小。
如果| 𝜙₁| = 1，无论滞后与否，过去值的影响是恒定的。
如果| 𝜙₁| > 1，则随着步骤的推移，过去值的影响会影响当前值。

$|\phi_1|<1$时，为弱平稳过程。
当AR(1)满足弱平稳过程时，由于白噪声平均值为0，即$E[U_t]=0$，有：$$
E[Y_t]=E[\phi_0]+E[\phi_1Y_{t-1}]+E[U_t]
\\
=\phi_0+\phi_1 E[Y_{t-1}]+0
$$
即：$\mu=\phi_0+\phi_1\mu$，$\mu=\frac{\phi_0}{1-\phi_1}$.

对于协方差，有：$$
Y_t=\mu(1-\phi_1)+\phi_1Y_{t-1}+U_t
$$
$$
Y_t-\mu=\phi_1(Y_{t-1}-\mu)+U_t
$$
则有:$$
E[(Y_t-\mu)^2]=E[(\phi_1(Y_{t-1}-\mu)+U_t)^2]
\\
V[Y_t]=E[\phi_1^2(Y_{t-1}-\mu)^2+2U_t\phi_1(Y_{t-1}-\mu)+U_t^2]
\\
V[Y_t]=\phi_1^2V[Y_{t-1}]+\sigma^2
$$
即：$\gamma_0=\frac{\sigma^2}{1-\phi_1^2}$.
对于协方差，有：$$
E[(Y_t-\mu)(Y_{t-h}-\mu)]=E[(\phi_1(Y_{t-1}-\mu)+U_t)(Y_{t-h}-\mu)]
\\
Cov[Y_t,Y_{t-h}]=\phi_1Cov[Y_t,Y_{t-h}]+E[U_t(Y_{t-h}-\mu)]
\\
=\phi_1Cov[Y_t,Y_{t-h}]
$$
即：$\gamma_h=\phi_1\gamma_{h-1}=...=\phi_1^h\gamma_0=\phi_1^h\frac{\sigma^2}{1-\phi_1^2}$
类似地考虑AR(p)过程：$$
Y_t=c+\phi_1Y_{t-1}+\phi_2Y_{t-2}+...+\phi_pY_{t-p}+U_t
$$

### 2.3 移动平均线过程（MA）
MA(1)过程：
$$Y_t=\mu+U_t+\theta_1U_{t-1}$$
MA(1)过程由白噪声组成，其均值始终为µ。
方差：$$
V[Y_t]=(1+\theta_1^2)\sigma^2
$$
协方差：$$
Cov[Y_t,Y_{t-h}]=(1+\theta_1^2)\sigma^2, h=0
\\
=\theta_1\sigma^2, h=1
\\
=0, h>1
$$
与AR(1)过程相比，均值和方差似乎保持不变。随着参数值的增大，序列变得相对平滑。注意MA(1)过程和白噪声方差不同。
一般来说，MA(q)过程也是弱平稳的。
$$
Y_t=\mu+U_t+\theta_1U_{t-1}+...+\theta_qU_{t-q}
$$
有：$$
E[Y_t]=0
\\
V[Y_t]=(1+\theta_1^2+...+\theta_q^2)\sigma^2
\\
Cov[Y_t,Y_{t-h}]=(1+\theta_1^2+...+\theta_q^2)\sigma^2, h=0
\\
=(\theta_h+\theta_1\theta_{h+1}+...+\theta_{q-h}\theta_q)\sigma^2,1\leq h \leq q
\\
0, h>q
$$
### 2.4 自回归移动平均(ARMA)过程
ARMA(p,q)如下：$$
Y_t=c+\phi_1Y_{t-1}+...+\phi_pY_{t-p}+U_t+\phi_1U_{t-1}+...+\phi_qU_{t-q}
$$
参数p和q对应于AR和MA过程的参数。
由于MA过程总是具有弱平稳性，因此ARMA过程的弱平稳性取决于AR部分。

### 2.5 ARIMA过程
定义差分算子$\nabla$:$$
\nabla x_t=x_t-x_{t-1}
\\
\nabla^2x_t=\nabla\nabla x_t=(x_t-x_{t-1})-(x_{t-1}-x_{t-2})
$$
将ARIMA(p, d, q)过程定义为:$$
\nabla^dY_t=c+\phi_1\nabla^dY_{t-1}+...+\phi_p\nabla^dY_{t-p}+\theta_1\nabla^dU_{t-1}+...+\theta_q\nabla^dU_{t-q}
$$
p为AR过程的阶数，d为待微分的次数，q为MA过程的阶数。
> 用自相关函数(ACF)图确定MA过程的阶数(q)，用部分自相关函数(PACF)图确定AR过程的阶数(p).
```python
 # fit stepwise auto-ARIMA
 '''
 y_train:
 训练数据序列，应该是一个一维的时间序列数据（通常是 pandas 的 Series 或 NumPy 的数组）。
 start_p=1, start_q=1:
 初始的 AR (自回归) 和 MA (滑动平均) 阶数
max_p=3, max_q=3:
AR 和 MA 阶数的最大值
seasonal=False:
指定是否拟合季节性 ARIMA 模型
d=d:
差分次数 d,这个参数用于使序列平稳
stepwise=True:
使用逐步搜索算法来选择模型。逐步搜索是一种启发式方法
 '''
 arima = pm.auto_arima(y_train, start_p=1, start_q=1,
                              max_p=3, max_q=3, # m=12,
                              seasonal=False,
                              d=d, trace=True,
                              error_action='ignore', # don't want to know if an order does not work
                              suppress_warnings=True, # don't want convergence warnings
                              stepwise=True) # set to stepwise
 arima.summary()
```

## 3. 时间序列的统计检验
### 3.1 增强Dickey-Fuller(ADF)检验
评估在给定的单变量时间序列中是否存在单位根。单位根的存在表示时间序列是非平稳的，即它的统计特性（如均值和方差）随时间变化。
$$
\Delta Y_t=c+\rho Y_{t-1}+\psi_1\Delta Y_{t-1}+...+\psi_{p-1}\Delta Y_{t-p+1}+U_t
$$
其中：$$
\rho=\phi_1+\phi_2+...+\phi_p-1
\\
\psi_j=-\sum_{k=j+1}^{p}\phi_k(j=1,2,...,p-1)
$$
零假设$H_0:\rho=0$,时间序列存在单位根，即序列是非平稳的。
备择假设$H_1:\rho<0$,时间序列不存在单位根，即序列是平稳的。

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# 生成示例时间序列数据
np.random.seed(0)
n = 100
t = np.arange(n)
y = np.cumsum(np.random.randn(n)) # 随机游走过程，通常非平稳

# 创建一个DataFrame
data = pd.DataFrame(y, columns=['Value'])

# 绘制时间序列图
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title('Time Series Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# 执行ADF检验
result = adfuller(data['Value'], autolag='AIC') # autolag='AIC' 选择最优滞后阶数

# 打印ADF检验结果
print('ADF Statistic:', result[0])
print('p-value:', result[1])
for key, value in result[4].items():
    print(f'Critical Value ({key}): {value}')
print('Lags Used:', result[2])
print('Number of Observations Used:', result[3])

# 结果解释
if result[1] < 0.05:
print('Reject the null hypothesis: The time series is likely stationary.')
else:
print('Fail to reject the null hypothesis: The time series is likely non-stationary.')

```

### 3.2 Durbin-Watson检验
评价时间序列回归模型中残差项是否具有自相关性。其统计量的值范围在0到4之间：
* DW接近2：表示残差之间没有自相关，即残差序列是独立的。
* DW接近0：表示正自相关，即相邻残差正相关。
* DW接近4：表示负自相关，即相邻残差负相关。

Durbin-Watson统计量公式：
$$
DW=\frac{\sum_{t=1}^{n-1}(e_t-e_{t-1})^2}{\sum_{t=1}^n e_t^2}
$$
其中，$e_t$是第t个残差，n是样本量。
```python
 from statsmodels.stats.stattools import durbin_watson

 arima = pm.arima.ARIMA(order=(2,1,2))
 arima.fit(y_train)

 dw = durbin_watson(arima.resid())
 print('DW statistic: ', dw)
```