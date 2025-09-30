import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ================= 数据读取 =================
df = pd.read_excel(
    r'C:\Users\鲁迅先生\Desktop\作业\pyth\@Python大数据分析与机器学习商业案例实战\@Python大数据分析与机器学习商业案例实战\第3章 线性回归模型\源代码汇总_PyCharm格式\客户价值数据表.xlsx'
)

# 特征列
X = df[['历史贷款金额', '贷款次数', '学历', '月收入', '性别']].values
y = df['客户价值'].values

n = len(y)

# ================== 1. 线性回归 ==================
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

lin_reg = LinearRegression()
lin_reg.fit(X, y)
r2_lin = lin_reg.score(X, y)
adj_r2_lin = 1 - (1 - r2_lin) * (n - 1) / (n - X.shape[1] - 1)

X_sm = sm.add_constant(X)
ols_model = sm.OLS(y, X_sm).fit()
p_values_lin = ols_model.pvalues

# ================== 2. 二次多项式回归 ==================
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

r2_poly = poly_reg.score(X_poly, y)
adj_r2_poly = 1 - (1 - r2_poly) * (n - 1) / (n - X_poly.shape[1] - 1)

ols_poly = sm.OLS(y, sm.add_constant(X_poly)).fit()
p_values_poly = ols_poly.pvalues

# ================== 3. 岭回归 ==================
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X, y)
r2_ridge = ridge_reg.score(X, y)
adj_r2_ridge = 1 - (1 - r2_ridge) * (n - 1) / (n - X.shape[1] - 1)

# ================== 4. 支持向量回归 ==================
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr_rbf.fit(X_scaled, y_scaled)
r2_svr = svr_rbf.score(X_scaled, y_scaled)
adj_r2_svr = 1 - (1 - r2_svr) * (n - 1) / (n - X.shape[1] - 1)

# ================== 5. KNN 回归 ==================
from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X, y)
r2_knn = knn_reg.score(X, y)
adj_r2_knn = 1 - (1 - r2_knn) * (n - 1) / (n - X.shape[1] - 1)

# ================== 结果汇总 ==================
results = pd.DataFrame({
    "模型": ["线性回归","二次多项式回归","岭回归","SVR (RBF)","KNN回归 (k=3)"],
    "R²": [r2_lin, r2_poly, r2_ridge, r2_svr, r2_knn],
    "调整后R²": [adj_r2_lin, adj_r2_poly, adj_r2_ridge, adj_r2_svr, adj_r2_knn]
})

print("\n=== 模型评估汇总 ===")
print(results)

# ================== 可视化 ==================
x = np.arange(len(results))
width = 0.35
colors = ['steelblue','cornflowerblue']

fig, ax = plt.subplots(figsize=(10,6))
bar1 = ax.bar(x - width/2, results['R²'], width, label='R方', color=colors[0], edgecolor='none')
bar2 = ax.bar(x + width/2, results['调整后R²'], width, label='调整后R方', color=colors[1], edgecolor='none')

# 显示数值
for bar in bar1 + bar2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0,3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

# 设置刻度和标签
ax.set_xticks(x)
ax.set_xticklabels(results['模型'], rotation=20, ha='right')
ax.set_ylabel("R方")
ax.set_title("客户价值预测模型对比 — R方与调整后R方")
ax.legend()
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()
