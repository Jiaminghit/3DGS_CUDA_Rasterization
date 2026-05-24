## gaussian splatting backward 的近似步骤:
### Overview
* Input : 
  | 输入 | 名称 | 维度|
  |-|-|-|
  | $point_{gaussian3d}$ | 3D 位置 | $(Gaussians, 3)$ |
  | $RGB_{gaussian3d}$ | 高斯椭球的颜色 | $(Gaussians, 3)$ |
  | $rotation_{gaussian3d}$ | 旋转四元数 | $(Gaussians, 4)$ |
  | $scale_{gaussian3d}$ | 椭球轴的长度 | $(Gaussians, 3)$ |
  | $opacity_{gaussian3d}$ | 椭球透明度 | $(Gaussians, 1)$ |
* Output :
  | 输出 | 名称 | 维度 |
  |-|-|-|
  | $RGB_{pixel}$ | 像素的颜色 | $(Pixles, 3)$ |
* Loss Function :
  $$L1 + SSIM$$
* **策略：** 由于 ```diff-gaussian-rasterization``` 分为两个过程：EWA Splatting + rendering，所以我们也需要分开进行反向传播即先反向传播rendering部分再反向传播EWA Splatting。
### renderCUDA 部分的梯度计算 —— 链式法则
#### 前置准备工作
* 已知 : $\frac{\partial Loss}{\partial {RGB_{pixel}}} $
  |已知| 求解 |
  |-|-|
  |$\frac{\partial Loss}{\partial {RGB_{pixel}}} $| $\frac{\partial Loss}{\partial RGB_{gaussian2d}} $ |
  || $\frac{\partial Loss}{\partial opacity_{gaussian2d}} $ |
  || $\frac{\partial Loss}{\partial \Sigma^{-1}_{gaussian2d}} $ |
  || $\frac{\partial Loss}{\partial \mu_{gaussian2d}}$ |
* 用到的重要前向渲染公式：
  1. **相对坐标计算 (Delta)：** 计算像素坐标 $(x, y)$ 与高斯球 2D 中心 $\mu = (\mu_x, \mu_y)$ 的差值。
    $$
    \begin{equation}
    \begin{aligned}
      dx &= x - \mu_x \\
      dy &= y - \mu_y
    \end{aligned}
    \end{equation}
    $$
  2. **高斯指数部分 ```(Power / G)```：** 利用 2D 协方差矩阵的逆（即 conic2D，包含三个独立元素 $\Sigma^{-1}_{11}, \Sigma^{-1}_{12}, \Sigma^{-1}_{22}$）计算马氏距离的负半值。
    $$
    \begin{equation}
    \begin{aligned}
      Power &= -\frac{1}{2} (X - \mu)^T \Sigma^{-1} (X - \mu) \\
            &= -0.5 \cdot \Sigma^{-1}_{11} \cdot dx^2 - \Sigma^{-1}_{12} \cdot dx \cdot dy - 0.5 \cdot \Sigma^{-1}_{22} \cdot dy^2 \\
            &= -0.5 \cdot \Sigma^{-1}_{11} \cdot (x - \mu_x)^2 - \Sigma^{-1}_{12} \cdot (x - \mu_x) \cdot (y - \mu_y) - 0.5 \cdot \Sigma^{-1}_{22} \cdot (y - \mu_y)^2
    \end{aligned}
    \end{equation}
    $$
  3. **当前层的最终 Alpha ($\alpha_i$)：** 由基础不透明度（opacity）乘上高斯衰减。
    $$
    \begin{equation}
    \begin{aligned}
      \alpha_i = opacity_i \cdot \exp(Power)
    \end{aligned}
    \end{equation}
    $$   
  4. **Alpha 混合与透射率 (Alpha-compositing)：** 设 $T_i$ 为光线到达第 $i$ 个高斯球时的累积透射率（即背景光还能透过多少，初始为 1）。最终像素颜色:
    $$
    \begin{equation}
    \begin{aligned}
      C_{pixel} &= \sum_{i} c_i \cdot \alpha_i \cdot T_i \\
                &= \sum_{i} c_i \cdot \alpha_i \cdot (1 - \alpha_0)(1 - \alpha_1)\dots(1 - \alpha_{i - 1})
    \end{aligned}
    \end{equation}
    $$    
#### 求解1：颜色梯度 $\frac{\partial Loss}{\partial RGB_{gaussian2d}} $ (对应了```renderCUDA```核函数中的```dL_dcolors```)
  $$
  \begin{equation}
  \begin{aligned}
    \frac{\partial Loss}{\partial RGB_{gaussian2d}}
              &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot{\frac{\partial {RGB_{pixel}}}{\partial RGB_{gaussian2d}}} \\
              &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot (\alpha_i \cdot T_i)
  \end{aligned}
  \end{equation}
  $$  
#### 求解2：基础不透明度 $ opacity_i $ 梯度 $\frac{\partial Loss}{\partial opacity_{gaussian2d}}$ (对应了```renderCUDA```核函数中的```dL_dopacity```)
  $$
  \begin{equation}
  \begin{aligned}
    \frac{\partial Loss}{\partial opacity_{gaussian2d}}
              &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot \frac{\partial {RGB_{pixel}}}{\partial opacity_i} \\
              &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot \frac{\partial {RGB_{pixel}}}{\partial \alpha_i} \cdot \frac{\partial{\alpha_i}}{\partial{opacity_i}} \\
              &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot \exp(Power) \cdot \frac{\partial {RGB_{pixel}}}{\partial \alpha_i} \\
              &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot \exp(Power) \cdot[T_i \cdot C_i - T_i \cdot \alpha_{i+1} \cdot C_{i+1} - T_i \cdot (1-\alpha_{i+1})\cdot \alpha_{i+2} \cdot C_{i+2} - \dots] \\
              &= \frac{\partial Loss}{\partial RGB_{pixel}} \cdot \exp(Power) \cdot T_i \cdot (c_i - C_{after\_norm})
  \end{aligned}
  \end{equation}
  $$ 
  > 针对梯度计算中的$\frac{\partial {RGB_{pixel}}}{\partial \alpha_i}$，我们可以通过将椭球分为前、中、后三部分得以简化计算，方法如下：
  > 由于 $$
  \begin{equation}
  \begin{aligned}
    C_{pixel} &= \sum_{i} c_i \cdot \alpha_i \cdot T_i \\
              &= \sum_{i = 0}^{k-1} c_i \cdot \alpha_i \cdot T_i  + T_k \cdot \alpha_k \cdot c_k + \sum_{i = k+1} c_i \cdot \alpha_i \cdot T_i\\
              &= C_{before} + T_k \cdot \alpha_k \cdot c_k + T_k \cdot (1 - \alpha_k) \cdot C_{after\_norm}
  \end{aligned}
  \end{equation}
  $$
  > $C_{after\_norm}$ 为后续所有高斯球在该点剥离了 $T_{i+1}$ 衰减后的归一化累积颜色。
  > $$
  \begin{equation}
  \begin{aligned}
    \frac{\partial {RGB_{pixel}}}{\partial \alpha_i} &= 
    \frac{\partial C_{pixel}}{\partial \alpha_i} = 0 + T_i \cdot c_i - T_i \cdot C_{after\_norm} = T_i \cdot (c_i - C_{after\_norm})
  \end{aligned}
  \end{equation}
  $$


#### 求解3：2D 协方差梯度 $\frac{\partial Loss}{\partial \Sigma^{-1}_{gaussian2d}}$ (对应了```renderCUDA```核函数中的```dL_dconic2D```)
  $$
  \begin{equation}
  \begin{aligned}
    \frac{\partial Loss}{\partial \Sigma^{-1}_{gaussian2d}} 
    &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot \frac{\partial {RGB_{pixel}}}{\partial \Sigma^{-1}_{gaussian2d}} \\
    &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot \frac{\partial {RGB_{pixel}}}{\partial \alpha_i} \cdot \frac{\partial \alpha_i}{\partial \Sigma^{-1}_{gaussian2d}} \\
    &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot \frac{\partial {RGB_{pixel}}}{\partial \alpha_i} \cdot \frac{\partial \alpha_i}{\partial Power} \cdot \frac{\partial Power }{\partial \Sigma^{-1}_{gaussian2d}} \\
    &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot \frac{\partial {RGB_{pixel}}}{\partial \alpha_i} \cdot \frac{\partial \alpha_i}{\partial Power} \cdot \frac{\partial (-0.5 \cdot \Sigma^{-1}_{11} \cdot dx^2 - \Sigma^{-1}_{12} \cdot dx \cdot dy - 0.5 \cdot \Sigma^{-1}_{22} \cdot dy^2) }{\partial 
    \begin{bmatrix}
      \Sigma^{-1}_{11} & \Sigma^{-1}_{12} \\
      \Sigma^{-1}_{21} & \Sigma^{-1}_{22}
    \end{bmatrix}} \\
    &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot  Opacity_i \cdot \exp(Power) \cdot 
    \begin{bmatrix}
      -0.5 (dx)^2 & -dxdy \\
      -dxdy & -0.5(dy)^2
    \end{bmatrix}
    \cdot \frac{\partial {RGB_{pixel}}}{\partial \alpha_i}
  \end{aligned}
  \end{equation}
  $$ 
#### 求解4：2D 均值坐标梯度 $\frac{\partial Loss}{\partial \mu_{gaussian2d}}$ (对应了```renderCUDA```核函数中的```dL_dmean2D```)
  $$
  \begin{equation}
  \begin{aligned}
    \frac{\partial Loss}{\partial \mu_{gaussian2d}}
    &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot \frac{\partial {RGB_{pixel}}}{\partial \alpha_i} \cdot \frac{\partial \alpha_i}{\partial Power} \cdot \frac{\partial Power}{\partial \mu_{gaussian2d}} \\
    &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot Opacity_i \cdot \exp(Power) \cdot \frac{\partial (-0.5 \cdot \Sigma^{-1}_{11} \cdot dx^2 - \Sigma^{-1}_{12} \cdot dx \cdot dy - 0.5 \cdot \Sigma^{-1}_{22} \cdot dy^2)}{\partial 
    \begin{bmatrix}
      \mu_x \\ \mu_y
    \end{bmatrix}
    } \cdot \frac{\partial {RGB_{pixel}}}{\partial \alpha_i} \\
    &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot Opacity_i \cdot \exp(Power) \cdot \frac{\partial [-0.5 \cdot \Sigma^{-1}_{11} \cdot (x - \mu_x)^2 - \Sigma^{-1}_{12} \cdot (x - \mu_x) \cdot (y - \mu_y) - 0.5 \cdot \Sigma^{-1}_{22} \cdot (y - \mu_y)^2]}{\partial 
    \begin{bmatrix}
      \mu_x \\ \mu_y
    \end{bmatrix}
    } \cdot \frac{\partial {RGB_{pixel}}}{\partial \alpha_i} \\
    &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot Opacity_i \cdot \exp(Power) \cdot 
    \begin{bmatrix}
      \Sigma^{-1}_{11} & \Sigma^{-1}_{12} \\
      \Sigma^{-1}_{12} & \Sigma^{-1}_{22}
    \end{bmatrix}
    \cdot 
    \begin{bmatrix}
      x - \mu_x \\
      y - \mu_y
    \end{bmatrix}
    \cdot \frac{\partial {RGB_{pixel}}}{\partial \alpha_i}
  \end{aligned}
  \end{equation}
  $$  

### preprocess 部分的梯度计算 —— 链式法则
#### 前置准备工作
* 
  | 已知 | 物理含义 | 对应CUDA Kernel函数中的参数 |
  |-|-|-|
  | $\frac{\partial Loss}{\partial RGB_{gaussian2d}} $ | 颜色梯度 | ```dL_dcolors``` |
  | $\frac{\partial Loss}{\partial opacity_{gaussian2d}} $ | 高斯的基础不透明度梯度 | ```dL_dopacity``` |
  | $\frac{\partial Loss}{\partial \Sigma^{-1}_{gaussian2d}} $ | 2D协方差的逆矩阵梯度 | ```dL_dconic2D``` | 
  | $\frac{\partial Loss}{\partial \mu_{gaussian2d}}$ | 2D高斯的均值梯度| ```dL_dmean2D``` |

  | 求解 | 物理含义| 对应CUDA Kernel函数中的参数 |
  |-|-|-|
  |  $\frac{\partial Loss}{\partial SH}$ | 球谐函数系数梯度 | ```dL_dsh``` |
  | $\frac{\partial Loss}{\partial scale_{gaussian3d}}$ | 3D高斯的缩放因子 | ```dL_dscales``` |
  | $\frac{\partial Loss}{\partial quaternion_{gaussian3d}}$ | 3D高斯的旋转四元数| ```dL_drots``` |
  | $\frac{\partial L}{\partial \mu_{gaussian3d}}$| 3D高斯的均值坐标 | ```dL_dmeans``` |
* 用到的重要**前向预处理**公式：
  1. 根据**旋转四元数**和**缩放矩阵**构建**3D协方差矩阵**：
  * 缩放因子矩阵：
  $$ S = \begin{bmatrix}
    s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & s_z
  \end{bmatrix}
  $$
  * 旋转四元数$q = [r, x, y, z]$需要根据罗德里格公式转化为旋转矩阵：
  $$ R = \begin{bmatrix}
    1 - 2(y^2 + z^2) & 2(xy - zw) & 2(xz + yw) \\
    2(xy + zw) & 1-2(x^2+z^2) & 2(yz - xw) \\
    2(xz - yw) & 2(yz + xw) & 1 - 2(x^2 + y^2)
  \end{bmatrix}
  $$
  * 协方差矩阵：
  $$\begin{equation}
    \begin{aligned}
      \Sigma_{3D} &= M^TM \\ &= (SR)^T(SR)\\ &=R^TS^TSR\\ &= \begin{bmatrix}
        1 - 2(y^2 + z^2) & 2(xy + zw) & 2(xz - yw) \\
        2(xy - zw) & 1-2(x^2+z^2) & 2(yz + xw) \\
        2(xz + yw) & 2(yz - xw) & 1 - 2(x^2 + y^2)
      \end{bmatrix} \cdot \begin{bmatrix}
    s_x^2 & 0 & 0 \\ 0 & s_y^2 & 0 \\ 0 & 0 & s_z^2
  \end{bmatrix} \cdot \begin{bmatrix}
    1 - 2(y^2 + z^2) & 2(xy - zw) & 2(xz + yw) \\
    2(xy + zw) & 1-2(x^2+z^2) & 2(yz - xw) \\
    2(xz - yw) & 2(yz + xw) & 1 - 2(x^2 + y^2)
  \end{bmatrix}
    \end{aligned}
  \end{equation}
  $$ 
  2. 从**3D协方差到2D协方差**的**EWA Splat**的过程：
  * Jacobi矩阵：
  $$
  \begin{equation}
  \begin{aligned}
  J &= 
  \begin{bmatrix}
  \frac{\partial u}{\partial x_c} & \frac{\partial u}{\partial y_c} & \frac{\partial u}{\partial z_c} \\ 
  \frac{\partial v}{\partial x_c} & \frac{\partial v}{\partial y_c} & \frac{\partial v}{\partial z_c} \\ 
  0 & 0 & 0
  \end{bmatrix} \\
    &= 
  \begin{bmatrix}
  \frac{f_x}{z_c} & 0 & -f_x \frac{x_c}{z_c^2} \\ 
  0 & \frac{f_y}{z_c} & -f_y \frac{y_c}{z_c^2} \\ 
  0 & 0 & 0
  \end{bmatrix}
  \end{aligned}
  \end{equation}
  $$
  * EWA Splat公式：
  $$
  \begin{equation}
    \Sigma_{2D} = J W \Sigma_{3D} W^T J^T
  \end{equation}
  $$
  3. **3D均值**到**2D均值**的投影 (MVP变换)
  * 从世界坐标系变换到齐次裁剪空间
  $$\begin{equation}
    \begin{aligned}
      p_{hom} &= (p_x, p_y, p_z, p_w)^T &= P \cdot V \cdot (x, y, z, 1)^T
    \end{aligned}
  \end{equation}
  $$
    > GAMES101告诉我们 Project 变换一般是这样的，其将z的信息融合到了x和y中：
  $$\begin{bmatrix} p_x \\ p_y \\ p_z \\ p_w \end{bmatrix} =
    \begin{bmatrix}
    P_{00} & 0 & P_{02} & 0 \\
    0 & P_{11} & P_{12} & 0 \\
    0 & 0 & P_{22} & P_{23} \\
    0 & 0 & 1 & 0 
    \end{bmatrix}
    \begin{bmatrix} x \\ y \\ z \\ 1 
  \end{bmatrix}$$
  * 变换坐标为标准化设备坐标系(NDC, Normalized Device Coordinates)：为了产生“近大远小”的透视效果，必须将齐次坐标除以它的第四个分量 $p_w$（实质上代表了深度信息的某种变形）。
  $$x_{ndc} = \frac{p_x}{p_w}$$$$y_{ndc} = \frac{p_y}{p_w}$$
  * 视口变换 (Viewport Transformation / Pixel Space)：将 $[-1, 1]$ 的 NDC 坐标映射到真实的屏幕像素坐标 $\mu_{2D} = (u, v)$ 上。已知屏幕的宽度为 $W$，高度为 $H$。
  $$u = \frac{(x_{ndc} + 1) \cdot W - 1}{2}$$
  $$v = \frac{(y_{ndc} + 1) \cdot H - 1}{2}$$
  4. **球谐函数**到**颜色**
  * 视角方向 $dir$：$dir = \frac{\mu_{3D} - cam\_pos}{||\mu_{3D} - cam\_pos||}$
  * 颜色 $c$：$c = \sum_{l, m} SH_l^m \cdot Y_l^m(dir) + 0.5$  （$Y_l^m$ 是球谐基函数，仅与视角方向 $dir$ 有关）。
  > * $dir = (x, y, z)$：从相机指向高斯球中心的归一化视角方向向量。
  > * $sh[0] \dots sh[15]$：当前高斯球在某一颜色通道（例如红色 R 通道）下的 16 个可学习的球谐系数。最终的颜色值是各阶结果的累加。
> 🟢 第 0 阶 (Degree 0) —— 基础环境光 (1 个系数)这是全向的常数项（相当于基础颜色，不随视角变化）：
  $$C_{l=0} = C_0 \cdot sh[0]$$
  其中常数 $C_0 = 0.28209479177387814$。
> 🟡 第 1 阶 (Degree 1) —— 线性依赖 (3 个系数)引入基于 $x, y, z$ 单一坐标的线性视角依赖：
> $$C_{l=1} = -C_1 \cdot y \cdot sh[1] + C_1 \cdot z \cdot sh[2] - C_1 \cdot x \cdot sh[3]$$其中常数 $C_1 = 0.4886025119029199$。
> 🟠 第 2 阶 (Degree 2) —— 二次依赖 (5 个系数)引入坐标两两相乘的依赖，捕捉更复杂的高光和镜面反射：
$$\begin{aligned}
C_{l=2} &= C_{2,0} \cdot (x \cdot y) \cdot sh[4] \\
        &+ C_{2,1} \cdot (y \cdot z) \cdot sh[5] \\
        &+ C_{2,2} \cdot (2z^2 - x^2 - y^2) \cdot sh[6] \\
        &+ C_{2,3} \cdot (x \cdot z) \cdot sh[7] \\
        &+ C_{2,4} \cdot (x^2 - y^2) \cdot sh[8]
\end{aligned}$$这组常数对应源码中的 ```SH_C2[]``` 数组：
$C_{2,0} = 1.0925484305920792$
$C_{2,1} = -1.0925484305920792$
$C_{2,2} = 0.31539156525252005$
$C_{2,3} = -1.0925484305920792$
$C_{2,4} = 0.5462742152960396$
🔴 第 3 阶 (Degree 3) —— 三次依赖 (7 个系数)用于捕获极高频率的光照细节（如锐利的反射光边缘）：
$$\begin{aligned}
C_{l=3} &= C_{3,0} \cdot y(3x^2 - y^2) \cdot sh[9] \\
        &+ C_{3,1} \cdot (xyz) \cdot sh[10] \\
        &+ C_{3,2} \cdot y(4z^2 - x^2 - y^2) \cdot sh[11] \\
        &+ C_{3,3} \cdot z(2z^2 - 3x^2 - 3y^2) \cdot sh[12] \\
        &+ C_{3,4} \cdot x(4z^2 - x^2 - y^2) \cdot sh[13] \\
        &+ C_{3,5} \cdot z(x^2 - y^2) \cdot sh[14] \\
        &+ C_{3,6} \cdot x(x^2 - 3y^2) \cdot sh[15]
\end{aligned}$$这组常数对应源码中的 ```SH_C3[]``` 数组：
$C_{3,0} = -0.5900435899266435$
$C_{3,1} = 2.890611442640554$
$C_{3,2} = -0.4570457994644658$
$C_{3,3} = 0.3731763325901154$
$C_{3,4} = -0.4570457994644658$
$C_{3,5} = 1.445305721320277$
$C_{3,6} = -0.5900435899266435$
🎨 最终颜色映射 (Final Result)将上述四个阶次的结果全部累加后，还需要进行最终的偏置和截断操作：$$Color_{raw} = C_{l=0} + C_{l=1} + C_{l=2} + C_{l=3} + 0.5$$$$Color_{final} = \max(Color_{raw}, 0.0)$$