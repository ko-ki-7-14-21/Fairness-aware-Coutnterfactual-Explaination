A = np.array([
    [392, 343, 142, 133, 127],      # energy
    [6.1, 12.6, 12.2, 21.3, 22.5],   # protein
    [0.9, 1.7, 10.2, 5.9, 4.5]      # fat
])

b1 = np.array([8000, 415, 213.6])
b2 = np.array([7040, 600, 192.8])


c0 = np.array([50, 40, 90, 70, 100]) # 元の価格 (c_hat)
h = np.array([100, 100, 30, 10, 100])  # 在庫量

n = len(c0)
m = A.shape[0]
foods = ["米", "小麦", "卵", "鶏肉", "魚"]
K = 2
lambda_1 = 0.5
lambda_2 = 2.0

D=[[]for i in range(len(c0))]
for i in range(len(c0)):
    D[i]=lambda_1 * c0[i] 以上、lambda_2 * c0[i] 以下のすべての整数の集合

次の問題をgurobiで解く(計算時間上限120s)
\begin{align*}
    \min_{\bm{X},\bm{Y},\bm{c},\bm{U},\bm{v},v_{\text{ave}},v_{\text{max}},\bm{Z}} \quad & (1-\alpha-\beta) \frac{1}{n} \sum^n_{i=1} \Big{(}\frac{c_i - \hat{c_i}}{\hat{c_i}}\Big{)}^2 + \alpha \frac{1}{k} \sum^K_{k=1}(v_k - v_{\text{ave}})^2 + \beta v_{\max}^2 \\
    \text{s. t.} \quad & \bm{A}\bm{x}_k \ge \bm{b}_k, \quad \bm{x}_k \ge \bm{0}  \tag*{$(k=1,...,K)$} \\
    & \bm{A}^\top \bm{y}_k \le \bm{c}, \quad \bm{y}_k \ge \bm{0} \tag*{$(k=1,...,K)$} \\
    & \bm{b}_k^{\top} \bm{y}_k = \sum^n_{i=1}\sum^{|D|}_{j=1}d_{i,j}u_{k,i,j} \tag*{$(k=1,...,K)$} \\
    & v_k = \frac{\sum^n_{i=1}\sum^{|D|}_{j=1}d_{i,j}u_{k,i,j} - \hat{\bm{c}}^{\top} \hat{\bm{x}}_k}{\hat{\bm{c}}^{\top} \hat{\bm{x}}_k} \tag*{$(k=1,...,K)$} \\
    & v_{\text{max}} \ge 0 ,\quad v_{\text{max}}  \ge v_k \tag*{$(k=1,...,K)$} \\ 
    & \sum^K_{k=1} \bm{x}_k \le \bm{h} \\
    & c_i=\sum^{|D|}_{j=1}d_{i,j}z_{i,j} \tag*{$(i=1,...,n)$}\\
    & 0 \ge u_{k,i,j} \ge Mz_{i,j}, \quad x_{k,i} -M(1-z_{i,j}) \ge u_{k,i,j} \ge x_{k,i} \tag*{$(i=1,...,n),(k=1,...,K)$}\\
    & z_{i,j} \in \{0,1\}, \quad \sum^{|D|}_{j=1} z_{i,j}=1 \tag*{$(i=1,...,n)$}\\
    & D=[\lambda_1\hat{\bm{c}},\lambda_2\hat{\bm{c}}]
\end{align*}
計算中は30sごとに経過を出力

price_term = \sum^n_{i=1}((c[i]-c0[i])/c0[i])**2
fairness_term = \sum^K_{k=1}(v[k]-v_ave)**2
G = 1/n * price_term + 1/K * fairness_term + v_max ** 2

print("alpha：",alpha,"beta：",beta)
print("評価指標 G：",G)
print("price_term：",price_term)
print("fairness_term：",fairness_term)
print("v_max^2：",v_max ** 2)
print("計算時間：",t)
print("gap：",gap)

print("c=".c.X)
print("x[1]=",x[1].X)
print("x[2]=".x[2].X)