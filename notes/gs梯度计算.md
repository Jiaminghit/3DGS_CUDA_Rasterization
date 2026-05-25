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


#### 求解3：2D 协方差矩阵的逆的梯度 $\frac{\partial Loss}{\partial \Sigma^{-1}_{gaussian2d}}$ (对应了```renderCUDA```核函数中的```dL_dconic2D```)
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
  > 前向传播把 conic 作为 Rendering 阶段的直接输入参数，那么 Rendering 的反向传播自然也就只能算到 $\frac{\partial L}{\partial \Sigma_{gaussian2d}^{-1}}$ (dL_dconic2D) 为止了
#### 求解4：2D 均值坐标梯度 $\frac{\partial Loss}{\partial \mu_{gaussian2d}}$ (对应了```renderCUDA```核函数中的```dL_dmean2D```)
  $$
  \begin{equation}
  \begin{aligned}
    \frac{\partial Loss}{\partial \mu_{gaussian2dPIXEL}}
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
  这里所求梯度是损失函数关于像素空间均值坐标的梯度，为了对接上游```preprocessCUDA```中预期的NDC（标准化设备坐标）空间输入，还需要求一个对**像素坐标梯度**与**视口变换缩放因子**做哈达玛积（Hadamard product）
  $$
  \begin{equation}
  \begin{aligned}
    \frac{\partial Loss}{\partial \mu_{gaussian2d}} = \frac{\partial Loss}{\partial \mu_{gaussian2dNDC}}
    &= \frac{\partial Loss}{\partial \mu_{gaussian2dPIXEL}} \odot \begin{bmatrix} 0.5 W \\ 0.5 H \end{bmatrix}
  \end{aligned}
  \end{equation}
  $$
  > 从标准化设备坐标到像素坐标的视口变换 (Viewport Transformation / Pixel Space)：将 $[-1, 1]$ 的 NDC 坐标映射到真实的屏幕像素坐标 $\mu_{2D} = (u, v)$ 上。已知屏幕的宽度为 $W$，高度为 $H$。
  $$u = \frac{(x_{ndc} + 1) \cdot W - 1}{2}$$$$v = \frac{(y_{ndc} + 1) \cdot H - 1}{2}$$    
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
  * 旋转四元数$q = [r, x, y, z]$需要根据四元数-旋转矩阵的公式转化为旋转矩阵：
  $$ R = \begin{bmatrix}
    1 - 2(y^2 + z^2) & 2(xy - zr) & 2(xz + yr) \\
    2(xy + zr) & 1-2(x^2+z^2) & 2(yz - xr) \\
    2(xz - yr) & 2(yz + xr) & 1 - 2(x^2 + y^2)
  \end{bmatrix}
  $$
  * 协方差矩阵：
  $$\begin{equation}
    \begin{aligned}
      \Sigma_{3D} &= M^TM \\ &= (SR)^T(SR)\\ &=R^TS^TSR
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
  * 从世界坐标系变换到齐次裁剪空间坐标
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
  * 从齐次裁剪坐标到标准化设备坐标(NDC, Normalized Device Coordinates)：为了产生“近大远小”的透视效果，必须将齐次坐标除以它的第四个分量 $p_w$（实质上代表了深度信息的某种变形）。
  $$x_{ndc} = \frac{p_x}{p_w}$$$$y_{ndc} = \frac{p_y}{p_w}$$
  * 从标准化设备坐标到像素坐标的视口变换 (Viewport Transformation / Pixel Space)：将 $[-1, 1]$ 的 NDC 坐标映射到真实的屏幕像素坐标 $\mu_{2D} = (u, v)$ 上。已知屏幕的宽度为 $W$，高度为 $H$。
  $$u = \frac{(x_{ndc} + 1) \cdot W - 1}{2}$$
  $$v = \frac{(y_{ndc} + 1) \cdot H - 1}{2}$$
  4. **球谐函数**到**颜色**
  * 视角方向 $dir$：$dir = \frac{\mu_{3D} - cam\_pos}{||\mu_{3D} - cam\_pos||}$
  * 颜色 $c$：
  $$c = \sum_{l, m} SH_l^m \cdot Y_l^m(dir) + 0.5$$  
  > * $Y_l^m$ 是球谐基函数，仅与视角方向 $dir$ 有关。
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
#### 求解1：球谐函数系数梯度 $\frac{\partial Loss}{\partial SH}$ (对应了preprocessCUDA核函数中的```dL_dsh```)
#### 求解2：(中间过渡变量) 3D 协方差矩阵梯度 $\frac{\partial Loss}{\partial \Sigma_{3D}}$ (对应了preprocessCUDA核函数中的```dL_dcov3D```)
  **已知：** 我们手中已经有的是从render阶段反向传回的**2d协方差矩阵的逆矩阵的梯度** $\frac{\partial Loss}{\partial \Sigma^{-1}_{gaussian2d}} $
  > Q : 为什么这里是逆矩阵的梯度？
  > A : 2D 高斯分布的概率密度函数：
  > $$G(x) \propto \exp\left(-\frac{1}{2} (x - \mu)^T \Sigma_{2D}^{-1} (x - \mu)\right)$$ 在 Rendering 的像素级循环中，我们要计算的是高斯衰减指数 Power。这个公式里天生自带的就是 $\Sigma_{2D}^{-1}$。如果我们求对 $\Sigma_{2D}$ 的梯度，在链式法则中就会多出求逆矩阵导数这一极其繁琐的步骤。**从工程角度看，求矩阵的逆是非常耗时的操作。** 如果每个像素在计算时都去对 $\Sigma_{2D}$ 求一次逆，GPU 会慢到无法忍受。因此，3DGS 在 preprocess 阶段，为每个高斯球计算好 $\Sigma_{2D}$ 后，立刻求逆，并把逆矩阵 $\Sigma_{2D}^{-1}$ 命名为 conic 保存下来。这样在渲染时，成千上万的像素只需要做简单的乘法和加法。
  
##### 第一步：从 逆矩阵 回退到 正矩阵 **($\frac{\partial Loss}{\partial \Sigma^{-1}_{gaussian2d}} \rightarrow \frac{\partial Loss}{\partial \Sigma_{gaussian2d}}$)**
  > **矩阵求导：**
  > 在矩阵微积分中，对于任意可逆对称矩阵 $A$，其逆矩阵的微分公式为：$d(A^{-1}) = -A^{-1} (dA) A^{-1}$
  > 代入 $A = \Sigma_{2D}$，已知目标是根据链式法则求 $\frac{\partial L}{\partial \Sigma_{2D}}$。将上述微分公式代入多元微积分的迹（Trace）技巧中，可以非常快地得出协方差矩阵梯度的转换公式：
  > $$\frac{\partial L}{\partial \Sigma_{2D}} = - \Sigma_{2D}^{-1} \cdot \left( \frac{\partial L}{\partial \Sigma_{2D}^{-1}} \right) \cdot \Sigma_{2D}^{-1}$$
  > **结论：** 误差对正矩阵的梯度，等于误差对逆矩阵的梯度在两边各乘一次逆矩阵，并反转方向。
  
  但是在 GPU 核函数（computeCov2DCUDA）中，**线程处理 $2 \times 2$ 矩阵的乘法非常浪费寄存器**。所以作者把这个 $2 \times 2$ 的对称矩阵拆成了 3 个独立的标量来进行链式求导。
  设 2D 协方差矩阵 $ \Sigma_{2D} = \begin{bmatrix} a & b \\ b & c \end{bmatrix} $
  $$\begin{equation}
    \begin{aligned}
      \Sigma_{2D}^{-1} &= \frac{1}{det} \begin{bmatrix} c & -b \\ -b & a \end{bmatrix} \\ 
      &= \begin{bmatrix} \frac{c}{ac - b^2} & \frac{-b}{ac - b^2} \\ \frac{-b}{ac - b^2} & \frac{a}{ac - b^2} \end{bmatrix} \\
      &= \begin{bmatrix} c_x & c_y \\ c_y & c_z \end{bmatrix}
    \end{aligned}
  \end{equation}
  $$  
  故有对应关系为：
  $$c_x = \frac{c}{a \cdot c - b^2}$$ $$c_y = \frac{-b}{a \cdot c - b^2}$$ $$c_z = \frac{a}{a \cdot c - b^2}$$
  $$
  \begin{equation}
    \begin{aligned}
      \frac{\partial c_x}{\partial a} &= \frac{0 \cdot det - c \cdot (c)}{det^2} = -\frac{c^2}{det^2} = -(c_x)^2 \\
      \frac{\partial c_y}{\partial a} &= \frac{0 \cdot det - (-b) \cdot (c)}{det^2} = \frac{b \cdot c}{det^2} = -(\frac{c}{det} \cdot \frac{-b}{det}) = -c_x \cdot c_y \\
      \frac{\partial c_z}{\partial a} &= \frac{1 \cdot det - a \cdot (c)}{det^2} = \frac{(ac - b^2) - ac}{det^2} = \frac{-b^2}{det^2} = -(c_y)^2 \\
      \frac{\partial c_x}{\partial b} &= \frac{0 \cdot det - (-2b) \cdot (c)}{det^2} = - 2\frac{-b}{det} \cdot \frac{c}{det}= -2 c_x c_y \\
      \frac{\partial c_y}{\partial b} &= \frac{-(ac - b^2) - (-2b\cdot(-b))}{det^2} = \frac{-ac - b^2}{det^2} = -c_x \cdot c_z - c_y^2 \\
      \frac{\partial c_z}{\partial b} &= \frac{0 \cdot det - (-2b) \cdot (a)}{det^2} = - 2\frac{-b}{det} \cdot \frac{a}{det}= -2 c_z c_y \\
      \frac{\partial c_x}{\partial c} &= \frac{ac - b^2 -ac}{det^2} = - \frac{-b}{det} \cdot \frac{-b}{det}= - c_y^2 \\
      \frac{\partial c_y}{\partial c} &= \frac{0 \cdot det - a\cdot(-b)}{det^2} = -c_y \cdot c_z \\
      \frac{\partial c_z}{\partial c} &= \frac{0 \cdot det - a \cdot a}{det^2} = - 2\frac{a}{det} \cdot \frac{a}{det}= - c_z^2
    \end{aligned}
  \end{equation}
  $$
  将损失函数对于矩阵的导数转化成为其对于矩阵元素的导数，进而将矩阵的链导法则转化为标量的链导法则：
  $$\begin{equation}
    \begin{aligned}
      \frac{\partial Loss}{\partial \Sigma_{gaussian2d}} &= \begin{bmatrix}
        \frac{\partial Loss}{\partial a} & \frac{\partial Loss}{\partial b} \\
        \frac{\partial Loss}{\partial b} & \frac{\partial Loss}{\partial c}
      \end{bmatrix} \\
      &= \begin{bmatrix}
        \frac{\partial L}{\partial c_x}\frac{\partial c_x}{\partial a} + \frac{\partial L}{\partial c_y}\frac{\partial c_y}{\partial a} + \frac{\partial L}{\partial c_z}\frac{\partial c_z}{\partial a} & 
        \frac{\partial L}{\partial c_x}\frac{\partial c_x}{\partial b} + \frac{\partial L}{\partial c_y}\frac{\partial c_y}{\partial b} + \frac{\partial L}{\partial c_z}\frac{\partial c_z}{\partial b} \\
        \frac{\partial L}{\partial c_x}\frac{\partial c_x}{\partial b} + \frac{\partial L}{\partial c_y}\frac{\partial c_y}{\partial b} + \frac{\partial L}{\partial c_z}\frac{\partial c_z}{\partial b} &
        \frac{\partial L}{\partial c_x}\frac{\partial c_x}{\partial c} + \frac{\partial L}{\partial c_y}\frac{\partial c_y}{\partial c} + \frac{\partial L}{\partial c_z}\frac{\partial c_z}{\partial c} 
      \end{bmatrix} \\
      &= \begin{bmatrix}
        \frac{\partial L}{\partial c_x}\cdot(-(c_x)^2) + \frac{\partial L}{\partial c_y}\cdot (-c_x \cdot c_y ) + \frac{\partial L}{\partial c_z}\cdot(-(c_y)^2) & 
        \frac{\partial L}{\partial c_x} \cdot (-2 c_x c_y)  + \frac{\partial L}{\partial c_y} \cdot (-c_x \cdot c_z - c_y^2) + \frac{\partial L}{\partial c_z} \cdot (-2 c_z c_y)\\
        \frac{\partial L}{\partial c_x} \cdot (-2 c_x c_y)  + \frac{\partial L}{\partial c_y} \cdot (-c_x \cdot c_z - c_y^2) + \frac{\partial L}{\partial c_z} \cdot (-2 c_z c_y)&
        \frac{\partial L}{\partial c_x} \cdot (- c_y^2) + \frac{\partial L}{\partial c_y} \cdot (-c_y \cdot c_z ) + \frac{\partial L}{\partial c_z} \cdot (- c_z^2)
      \end{bmatrix}
    \end{aligned}
  \end{equation}
  $$
  > $\frac{\partial L}{\partial c_x} = \frac{\partial Loss}{\partial \Sigma^{-1}_{gaussian2d}}$ 矩阵的左上角元素 
  $\frac{\partial L}{\partial c_y} = \frac{\partial Loss}{\partial \Sigma^{-1}_{gaussian2d}}$ 矩阵的右上角元素 \ 左下角元素
  $\frac{\partial L}{\partial c_z} = \frac{\partial Loss}{\partial \Sigma^{-1}_{gaussian2d}}$ 矩阵的右下角元素

##### 第二步：从 2D高斯 回退到 3D高斯 **($\frac{\partial Loss}{\partial \Sigma_{gaussian2d}} \rightarrow \frac{\partial Loss}{\partial \Sigma_{gaussian3d}}$)**
  由于：$$\Sigma_{gaussian2d} = T \Sigma_{gaussian3d} T^T = J \cdot W \cdot \Sigma_{gaussian3d} \cdot W^T \cdot J^T$$
  根据矩阵微分的 Trace ：
  $$\begin{equation}
    \begin{aligned}
      \frac{\partial Loss}{\partial \Sigma_{gaussian3d}} &= T^T \frac{\partial L}{\partial \Sigma_{gaussian2d}} T \\
      &= W^T \cdot J^T \cdot \frac{\partial L}{\partial \Sigma_{gaussian2d}} J \cdot W \\
      &= W^T \cdot J^T \cdot \begin{bmatrix}
        \frac{\partial Loss}{\partial a} & \frac{\partial Loss}{\partial b} & 0\\
        \frac{\partial Loss}{\partial b} & \frac{\partial Loss}{\partial c} & 0\\
        0 & 0 & 0
      \end{bmatrix} \cdot J  \cdot T
    \end{aligned}
  \end{equation}
  $$
  > 上式中都是$3 \times 3$的矩阵
#### 求解3：3D 缩放梯度 $\frac{\partial Loss}{\partial S}$ 
> ❗注意矩阵求导不能简单用标量的链导法则解决，其应遵循**全微分（Differential）和迹（Trace）技巧**。矩阵求导只有一个核心第一性原理：$$d(Loss) = \text{tr}\left( \left(\frac{\partial Loss}{\partial X}\right)^T dX \right)$$只要你能把等式右边凑成 $\text{tr}(\text{某矩阵}^T \cdot dX)$ 的形式，那个“某矩阵”就是完美的梯度！
##### 前半阶段：求$\frac{\partial Loss}{\partial M}$
  $$\begin{equation}
    \begin{aligned}
      已知:\\
      \Sigma_{gaussian3d} &= M^T \cdot M \\ 
      d\Sigma_{gaussian3d} &= d(M^T) M + M^T dM = (dM)^T M + M^T dM \\
      代入第一性原理:\\
      d(Loss) &= \text{tr}\left( (\frac{\partial Loss}{\partial \Sigma_{gaussian3d}})^T d\Sigma_{gaussian3d} \right) = \text{tr}\left( \frac{\partial Loss}{\partial \Sigma_{gaussian3d}} ( (dM)^T M + M^T dM ) \right) \\
      由于迹的性质:\\ \text{tr}(A) &= \text{tr}(A^T) \\ 
      \text{tr}(AB) &= \text{tr}(BA) \\ 
      \text{tr}(A + B) &= \text{tr}(A) + \text{tr}(B) \\
      所以可以简化上式为:\\ 
      \text{tr}(\frac{\partial Loss}{\partial \Sigma_{gaussian3d}} (dM)^T M) &= \text{tr}(M \frac{\partial Loss}{\partial \Sigma_{gaussian3d}} (dM)^T) \\
      &= \text{tr}( (dM (\frac{\partial Loss}{\partial \Sigma_{gaussian3d}})^T M^T)^T ) \\ 
      &= \text{tr}(dM \frac{\partial Loss}{\partial \Sigma_{gaussian3d}} M^T)\\ 
      &= \text{tr}(\frac{\partial Loss}{\partial \Sigma_{gaussian3d}} M^T dM) \\
      故合并得到:\\
      d(Loss) &= \text{tr}(\frac{\partial Loss}{\partial \Sigma_{gaussian3d}} M^T dM) + \text{tr}(\frac{\partial Loss}{\partial \Sigma_{gaussian3d}}  M^T dM ) \\
      &= \text{tr}(2\cdot\frac{\partial Loss}{\partial \Sigma_{gaussian3d}} M^T \cdot dM)\\
      &= \text{tr}((\frac{\partial Loss}{\partial M})^T \cdot dM) \\
      所以:\\
      \pmb{\frac{\partial Loss}{\partial M}} &= 2 \cdot M \cdot (\frac{\partial Loss}{\partial \Sigma_{gaussian3d}})^T \\
      &= \pmb{2 \cdot M \cdot \frac{\partial Loss}{\partial \Sigma_{gaussian3d}}} \\      
    \end{aligned}
  \end{equation}
  $$
##### 后半阶段：求$\frac{\partial Loss}{\partial S}$
  $$\begin{equation}
    \begin{aligned}
      已知:\\
      M &= S \cdot R \\
      dM &= dS \cdot R \\
      \frac{\partial Loss}{\partial M} &= 2 \cdot M \cdot \frac{\partial Loss}{\partial \Sigma_{gaussian3d}} \\
      带入第一性原理:\\
      d(Loss) &= \text{tr}\left( (\frac{\partial Loss}{\partial M})^T dM \right) = \text{tr}\left( (\frac{\partial Loss}{\partial M})^T (dS \cdot R) \right)\\
      &= \text{tr}\left( R \cdot (\frac{\partial Loss}{\partial M})^T \cdot dS \right)\\
      所以:\\
      (\frac{\partial Loss}{\partial S})^T 
      &= R \cdot (\frac{\partial Loss}{\partial M})^T\\
      \frac{\partial Loss}{\partial S} &= \frac{\partial Loss}{\partial M} \cdot R^T = 2 \cdot M \cdot \frac{\partial Loss}{\partial \Sigma_{gaussian3d}} \cdot R^T \\
      &= 2 \cdot S \cdot R \cdot \frac{\partial Loss}{\partial \Sigma_{gaussian3d}} \cdot R^T      
    \end{aligned}
  \end{equation}
  $$
##### 最后一步
  这里的缩放因子矩阵$S$是一个对角矩阵$S = \text{diag}(s_x, s_y, s_z)$。所以，我们只需要提取上述结果矩阵的主对角线元素即可，即 $$\frac{\partial Loss}{\partial S} = \text{diag}(2 \cdot S \cdot R \cdot \frac{\partial Loss}{\partial \Sigma_{gaussian3d}} \cdot R^T)$$
#### 求解4：3D 旋转梯度 $\frac{\partial Loss}{\partial q}$
##### 前半阶段，借助上一段求解出来的$\frac{\partial Loss}{\partial M}$求$\frac{\partial Loss}{\partial R}$：
  $$\begin{equation}
    \begin{aligned}
      已知:\\
      M &= S \cdot R \\
      dM &= S \cdot dR \\
      \frac{\partial Loss}{\partial M} &= 2 \cdot M \cdot \frac{\partial Loss}{\partial \Sigma_{gaussian3d}} \\
      带入第一性原理:\\
      d(Loss) &= \text{tr}\left( (\frac{\partial Loss}{\partial M})^T dM \right) = \text{tr}\left( (\frac{\partial Loss}{\partial M})^T (S \cdot dR) \right)\\
      &= \text{tr}\left((\frac{\partial Loss}{\partial M})^T \cdot S \cdot dR \right)\\
      所以:\\
      (\frac{\partial Loss}{\partial R})^T 
      &= (\frac{\partial Loss}{\partial M})^T \cdot S\\
      \frac{\partial Loss}{\partial R} &= S^T \cdot \frac{\partial Loss}{\partial M} = S^T \cdot 2 \cdot M \cdot \frac{\partial Loss}{\partial \Sigma_{gaussian3d}} \\
      &= S \cdot 2 \cdot S \cdot R \cdot \frac{\partial Loss}{\partial \Sigma_{gaussian3d}}   
    \end{aligned}
  \end{equation}
  $$
##### 后半阶段,分别求$\frac{\partial Loss}{\partial r}$、$\frac{\partial Loss}{\partial x}$、$\frac{\partial Loss}{\partial y}$、$\frac{\partial Loss}{\partial z}$
* 旋转四元数$q = [r, x, y, z]$与旋转矩阵$R$的映射关系：
  $$ R(q) = \begin{bmatrix}
    1 - 2(y^2 + z^2) & 2(xy - zr) & 2(xz + yr) \\
    2(xy + zr) & 1-2(x^2+z^2) & 2(yz - xr) \\
    2(xz - yr) & 2(yz + xr) & 1 - 2(x^2 + y^2)
  \end{bmatrix}
  $$
* 刚刚求出的误差对旋转矩阵的梯度矩阵：$$U = \frac{\partial Loss}{\partial R} = S \cdot 2 \cdot S \cdot R \cdot \frac{\partial Loss}{\partial \Sigma_{gaussian3d}} $$
* 根据标量对矩阵求导的链式法则（Frobenius 内积），可以分别求出：
  $$\begin{equation}
    \begin{aligned}
      \frac{\partial Loss}{\partial r} &= \sum_{i=1}^3 \sum_{j=1}^3 \frac{\partial Loss}{\partial R_{ij}} \cdot \frac{\partial R_{ij}}{\partial r} &= \sum_{i=1}^3 \sum_{j=1}^3 U_{ij} \cdot \frac{\partial R_{ij}}{\partial r} \\
      \frac{\partial Loss}{\partial x} &= \sum_{i=1}^3 \sum_{j=1}^3 \frac{\partial Loss}{\partial R_{ij}} \cdot \frac{\partial R_{ij}}{\partial x} &= \sum_{i=1}^3 \sum_{j=1}^3 U_{ij} \cdot \frac{\partial R_{ij}}{\partial x} \\
      \frac{\partial Loss}{\partial y} &= \sum_{i=1}^3 \sum_{j=1}^3 \frac{\partial Loss}{\partial R_{ij}} \cdot \frac{\partial R_{ij}}{\partial y} &= \sum_{i=1}^3 \sum_{j=1}^3 U_{ij} \cdot \frac{\partial R_{ij}}{\partial y} \\
      \frac{\partial Loss}{\partial z} &= \sum_{i=1}^3 \sum_{j=1}^3 \frac{\partial Loss}{\partial R_{ij}} \cdot \frac{\partial R_{ij}}{\partial z} &= \sum_{i=1}^3 \sum_{j=1}^3 U_{ij} \cdot \frac{\partial R_{ij}}{\partial z}
    \end{aligned}
  \end{equation}
  $$
  
  $$\begin{equation}
    \begin{aligned}
      \frac{\partial Loss}{\partial r} &= 2 \cdot (x U_{23} - x U_{32} + y U_{31} - y U_{13} + z U_{12} - z U_{21}) \\
      \frac{\partial Loss}{\partial x} &= 2 \cdot (-2x U_{22} - 2x U_{33} + y U_{12} + y U_{21} + z U_{13} + z U_{31} + r U_{23} - r U_{32}) \\
      \frac{\partial Loss}{\partial y} &= 2 \cdot (x U_{12} + x U_{21} - 2y U_{11} - 2y U_{33} + z U_{23} + z U_{32} + r U_{31} - r U_{13}) \\
      \frac{\partial Loss}{\partial z} &= 2 \cdot (x U_{13} + x U_{31} + y U_{23} + y U_{32} - 2z U_{11} - 2z U_{22} + r U_{12} - r U_{21})
    \end{aligned}
  \end{equation}
  $$