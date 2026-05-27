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
      \begin{aligned}
        dx &= x - \mu_x \\
        dy &= y - \mu_y
      \end{aligned}
      $$

  2. **高斯指数部分 ```(Power / G)```：** 利用 2D 协方差矩阵的逆（即 conic2D，包含三个独立元素 $\Sigma^{-1}_{11}, \Sigma^{-1}_{12}, \Sigma^{-1}_{22}$ ）计算马氏距离的负半值。
      
      $$
      \begin{aligned}
        Power &= -\frac{1}{2} (X - \mu)^T \Sigma^{-1} (X - \mu) \\
              &= -0.5 \cdot \Sigma^{-1}_{11} \cdot dx^2 - \Sigma^{-1}_{12} \cdot dx \cdot dy - 0.5 \cdot \Sigma^{-1}_{22} \cdot dy^2 \\
              &= -0.5 \cdot \Sigma^{-1}_{11} \cdot (x - \mu_x)^2 - \Sigma^{-1}_{12} \cdot (x - \mu_x) \cdot (y - \mu_y) - 0.5 \cdot \Sigma^{-1}_{22} \cdot (y - \mu_y)^2
      \end{aligned}
      $$

  3. **当前层的最终 Alpha ($\alpha_i$)：** 由基础不透明度（opacity）乘上高斯衰减。
      
      $$
      \begin{aligned}
        \alpha_i = opacity_i \cdot \exp(Power)
      \end{aligned}
      $$   
    
  4. **Alpha 混合与透射率 (Alpha-compositing)：** 设 $T_i$ 为光线到达第 $i$ 个高斯球时的累积透射率（即背景光还能透过多少，初始为 1）。最终像素颜色:

      $$
      \begin{aligned}
        C_{pixel} &= \sum_{i} c_i \cdot \alpha_i \cdot T_i \\
                  &= \sum_{i} c_i \cdot \alpha_i \cdot (1 - \alpha_0)(1 - \alpha_1)\dots(1 - \alpha_{i - 1})
      \end{aligned}
      $$    
#### 求解1：颜色梯度 $\frac{\partial Loss}{\partial RGB_{gaussian2d}}$ (对应了```renderCUDA```核函数中的```dL_dcolors```)

  $$
  \begin{aligned}
    \frac{\partial Loss}{\partial RGB_{gaussian2d}}
              &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot{\frac{\partial {RGB_{pixel}}}{\partial RGB_{gaussian2d}}} \\
              &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot (\alpha_i \cdot T_i)
  \end{aligned}
  $$ 

  将结果按照 **三个通道(R, G, B)** 进行展开：

  $$
  \begin{aligned}
    \frac{\partial Loss}{\partial RGB_{gaussian2d}}
              &= \begin{bmatrix}
                  \frac{\partial Loss}{\partial R_{gaussian2d}} \\
                  \frac{\partial Loss}{\partial G_{gaussian2d}} \\
                  \frac{\partial Loss}{\partial B_{gaussian2d}}
                  \end{bmatrix} 
              &= \begin{bmatrix}
                  \frac{\partial Loss}{\partial R_{pixel}} \\
                  \frac{\partial Loss}{\partial G_{pixel}} \\
                  \frac{\partial Loss}{\partial B_{pixel}}
                  \end{bmatrix}
                  \cdot (\alpha_i \cdot T_i)
  \end{aligned}
  $$     

#### 求解2：基础不透明度 $ opacity_i $ 梯度 $\frac{\partial Loss}{\partial opacity_{gaussian2d}}$ (对应了```renderCUDA```核函数中的```dL_dopacity```)
  
  $$
  \begin{aligned}
    \frac{\partial Loss}{\partial opacity_{gaussian2d}}
              &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot \frac{\partial {RGB_{pixel}}}{\partial opacity_i} \\
              &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot \frac{\partial {RGB_{pixel}}}{\partial \alpha_i} \cdot \frac{\partial{\alpha_i}}{\partial{opacity_i}} \\
              &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot \exp(Power) \cdot \frac{\partial {RGB_{pixel}}}{\partial \alpha_i} \\
              &= \frac{\partial Loss}{\partial {RGB_{pixel}}} \cdot \exp(Power) \cdot[T_i \cdot C_i - T_i \cdot \alpha_{i+1} \cdot C_{i+1} - T_i \cdot (1-\alpha_{i+1})\cdot \alpha_{i+2} \cdot C_{i+2} - \dots] \\
              &= \frac{\partial Loss}{\partial RGB_{pixel}} \cdot \exp(Power) \cdot T_i \cdot (c_i - C_{after\_norm})
  \end{aligned}
  $$ 

  > 针对梯度计算中的$\frac{\partial {RGB_{pixel}}}{\partial \alpha_i}$，我们可以通过将椭球分为前、中、后三部分得以简化计算，方法如下：
  > 由于 
  
  > $$
  \begin{aligned}
    C_{pixel} &= \sum_{i} c_i \cdot \alpha_i \cdot T_i \\
              &= \sum_{i = 0}^{k-1} c_i \cdot \alpha_i \cdot T_i  + T_k \cdot \alpha_k \cdot c_k + \sum_{i = k+1} c_i \cdot \alpha_i \cdot T_i\\
              &= C_{before} + T_k \cdot \alpha_k \cdot c_k + T_k \cdot (1 - \alpha_k) \cdot C_{after\_norm}
  \end{aligned}
  $$

  > $C_{after\_norm}$ 为后续所有高斯球在该点剥离了 $T_{i+1}$ 衰减后的归一化累积颜色。
  
  > $$
  \begin{aligned}
    \frac{\partial {RGB_{pixel}}}{\partial \alpha_i} &= 
    \frac{\partial C_{pixel}}{\partial \alpha_i} = 0 + T_i \cdot c_i - T_i \cdot C_{after\_norm} = T_i \cdot (c_i - C_{after\_norm})
  \end{aligned}
  $$


#### 求解3：2D 协方差矩阵的逆的梯度 $\frac{\partial Loss}{\partial \Sigma^{-1}_{gaussian2d}}$ (对应了```renderCUDA```核函数中的```dL_dconic2D```)
  
  $$
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
  $$ 
  
  > 前向传播把 conic 作为 Rendering 阶段的直接输入参数，那么 Rendering 的反向传播自然也就只能算到 $\frac{\partial L}{\partial \Sigma_{gaussian2d}^{-1}}$ (dL_dconic2D) 为止了
#### 求解4：2D 均值坐标梯度 $\frac{\partial Loss}{\partial \mu_{gaussian2d}}$ (对应了```renderCUDA```核函数中的```dL_dmean2D```)
  
  $$
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
  $$  

  这里所求梯度是损失函数关于像素空间均值坐标的梯度，为了对接上游```preprocessCUDA```中预期的NDC（标准化设备坐标）空间输入，还需要求一个对**像素坐标梯度**与**视口变换缩放因子**做哈达玛积（Hadamard product）
  
  $$
  \begin{aligned}
    \frac{\partial Loss}{\partial \mu_{gaussian2d}} = \frac{\partial Loss}{\partial \mu_{gaussian2dNDC}}
    &= \frac{\partial Loss}{\partial \mu_{gaussian2dPIXEL}} \odot \begin{bmatrix} 0.5 W \\ 0.5 H \end{bmatrix}
  \end{aligned}
  $$

  > 从标准化设备坐标到像素坐标的视口变换 (Viewport Transformation / Pixel Space)：将 $[-1, 1]$ 的 NDC 坐标映射到真实的屏幕像素坐标 $\mu_{2D} = (u, v)$ 上。已知屏幕的宽度为 $W$，高度为 $H$。
  
  $$
  u = \frac{(x_{ndc} + 1) \cdot W - 1}{2}
  $$
  
  $$
  v = \frac{(y_{ndc} + 1) \cdot H - 1}{2}
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
  $$
  * EWA Splat公式：
  
  $$
  \Sigma_{2D} = J W \Sigma_{3D} W^T J^T
  $$
  3. **3D均值**到**2D均值**的投影 (MVP变换)
  * 从世界坐标系变换到齐次裁剪空间坐标
  
  $$
    \begin{aligned}
      p_{hom} &= (p_x, p_y, p_z, p_w)^T &= P \cdot V \cdot (x, y, z, 1)^T
    \end{aligned}
  $$
    > GAMES101告诉我们 Project 变换一般是这样的，其将z的信息融合到了x和y中：
  
  > $$
  \begin{bmatrix} p_x \\ p_y \\ p_z \\ p_w \end{bmatrix} =
    \begin{bmatrix}
    P_{00} & 0 & P_{02} & 0 \\
    0 & P_{11} & P_{12} & 0 \\
    0 & 0 & P_{22} & P_{23} \\
    0 & 0 & 1 & 0 
    \end{bmatrix}
    \begin{bmatrix} x \\ y \\ z \\ 1 
  \end{bmatrix}
  $$
  * 从齐次裁剪坐标到标准化设备坐标(NDC, Normalized Device Coordinates)：为了产生“近大远小”的透视效果，必须将齐次坐标除以它的第四个分量 $p_w$（实质上代表了深度信息的某种变形）。
  
  $$
  x_{ndc} = \frac{p_x}{p_w}
  $$
  $$
  y_{ndc} = \frac{p_y}{p_w}
  $$
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
由于球谐函数到颜色的映射是一个针对各颜色通道**独立计算**的多项式线性组合，加之最后有一个防止负数的截断操作 $\max(Color_{raw}, 0.0)$，我们可以利用**链式法则**直接对该多项式逐项求导。对于任意一个颜色通道（以 $Color_{channel}$ 代表 R, G 或 B 通道）的第 $i$ 个球谐系数 $sh[i]$，其链式求导路径为：

$$
\begin{aligned}
  Channel \space R : \frac{\partial Loss}{\partial sh[i]} &= \frac{\partial Loss}{\partial Color_{final}} \cdot \frac{\partial Color_{final}}{\partial Color_{raw}} \cdot \frac{\partial Color_{raw}}{\partial sh[i]} \\
  &= \frac{\partial Loss}{\partial R_{gaussian2d}} \cdot \mathbb{I}(Color_{raw} > 0) \cdot Y_i(dir) \\
  Channel \space G : \frac{\partial Loss}{\partial sh[i]} &= \frac{\partial Loss}{\partial G_{gaussian2d}} \cdot \mathbb{I}(Color_{raw} > 0) \cdot Y_i(dir) \\
  Channel \space B : \frac{\partial Loss}{\partial sh[i]} &= \frac{\partial Loss}{\partial B_{gaussian2d}} \cdot \mathbb{I}(Color_{raw} > 0) \cdot Y_i(dir)
\end{aligned}
$$

由于 R、G、B 三个通道的球谐系数是完全独立的（每个通道各自拥有 16 个单独的 $sh$ 系数，总计 48 个），因此梯度计算无需跨通道累加。针对某单一通道的所有 16 个球谐系数梯度，可以表示为一个列向量：

$$
\begin{aligned}
  \frac{\partial Loss}{\partial SH_{channel}} = \frac{\partial Loss}{\partial Color_{channel}} \cdot \mathbb{I}(Color_{raw} > 0) \cdot 
  \begin{bmatrix}
    C_0 \\
    -C_1 \cdot y \\
    C_1 \cdot z \\
    -C_1 \cdot x \\
    C_{2,0} \cdot (x \cdot y) \\
    \dots \\
    C_{3,6} \cdot x(x^2 - 3y^2)
  \end{bmatrix}
\end{aligned}
$$

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
  
  $$
    \begin{aligned}
      \Sigma_{2D}^{-1} &= \frac{1}{det} \begin{bmatrix} c & -b \\ -b & a \end{bmatrix} \\ 
      &= \begin{bmatrix} \frac{c}{ac - b^2} & \frac{-b}{ac - b^2} \\ \frac{-b}{ac - b^2} & \frac{a}{ac - b^2} \end{bmatrix} \\
      &= \begin{bmatrix} c_x & c_y \\ c_y & c_z \end{bmatrix}
    \end{aligned}
  $$  

  故有对应关系为：
  
  $$
  c_x = \frac{c}{a \cdot c - b^2}
  $$ 
  $$
  c_y = \frac{-b}{a \cdot c - b^2}
  $$ 
  $$
  c_z = \frac{a}{a \cdot c - b^2}
  $$
  
  $$
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
  $$

  将损失函数对于矩阵的导数转化成为其对于矩阵元素的导数，进而将矩阵的链导法则转化为标量的链导法则：
  
  $$
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
  $$
  > $\frac{\partial L}{\partial c_x} = \frac{\partial Loss}{\partial \Sigma^{-1}_{gaussian2d}}$ 矩阵的左上角元素 
  > $\frac{\partial L}{\partial c_y} = \frac{\partial Loss}{\partial \Sigma^{-1}_{gaussian2d}}$ 矩阵的右上角元素 \ 左下角元素
  > $\frac{\partial L}{\partial c_z} = \frac{\partial Loss}{\partial \Sigma^{-1}_{gaussian2d}}$ 矩阵的右下角元素

##### 第二步：从 2D高斯 回退到 3D高斯 ($\frac{\partial Loss}{\partial \Sigma_{gaussian2d}}\rightarrow\frac{\partial Loss}{\partial \Sigma_{gaussian3d}}$)
  由于：
  $$
  \Sigma_{gaussian2d} = T \Sigma_{gaussian3d} T^T = J \cdot W \cdot \Sigma_{gaussian3d} \cdot W^T \cdot J^T
  $$
  根据矩阵微分的 Trace ：
  
  $$
    \begin{aligned}
      \frac{\partial Loss}{\partial \Sigma_{gaussian3d}} &= T^T \frac{\partial L}{\partial \Sigma_{gaussian2d}} T \\
      &= W^T \cdot J^T \cdot \frac{\partial L}{\partial \Sigma_{gaussian2d}} J \cdot W \\
      &= W^T \cdot J^T \cdot \begin{bmatrix}
        \frac{\partial Loss}{\partial a} & \frac{\partial Loss}{\partial b} & 0\\
        \frac{\partial Loss}{\partial b} & \frac{\partial Loss}{\partial c} & 0\\
        0 & 0 & 0
      \end{bmatrix} \cdot J  \cdot T
    \end{aligned}
  $$

  > 上式中都是$3 \times 3$的矩阵
#### 求解3：3D 缩放梯度 $\frac{\partial Loss}{\partial S}$ 
> ❗注意矩阵求导不能简单用标量的链导法则解决，其应遵循**全微分（Differential）和迹（Trace）技巧**。矩阵求导只有一个核心第一性原理：$$d(Loss) = \text{tr}\left( \left(\frac{\partial Loss}{\partial X}\right)^T dX \right)$$只要你能把等式右边凑成 $\text{tr}(\text{某矩阵}^T \cdot dX)$ 的形式，那个“某矩阵”就是完美的梯度！
##### 前半阶段：求$\frac{\partial Loss}{\partial M}$
  
  $$
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
  $$

##### 后半阶段：求$\frac{\partial Loss}{\partial S}$
  
  $$
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
  $$

##### 最后一步
  这里的缩放因子矩阵$S$是一个对角矩阵$S = \text{diag}(s_x, s_y, s_z)$。所以，我们只需要提取上述结果矩阵的主对角线元素即可，即 $$\frac{\partial Loss}{\partial S} = \text{diag}(2 \cdot S \cdot R \cdot \frac{\partial Loss}{\partial \Sigma_{gaussian3d}} \cdot R^T)$$
#### 求解4：3D 旋转梯度 $\frac{\partial Loss}{\partial q}$
##### 前半阶段，借助上一段求解出来的$\frac{\partial Loss}{\partial M}$求$\frac{\partial Loss}{\partial R}$：

  $$
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
  $$

##### 后半阶段,分别求$\frac{\partial Loss}{\partial r}$、$\frac{\partial Loss}{\partial x}$、$\frac{\partial Loss}{\partial y}$、$\frac{\partial Loss}{\partial z}$
* 旋转四元数$q = [r, x, y, z]$与旋转矩阵$R$的映射关系：
  
  $$ 
  R(q) = \begin{bmatrix}
    1 - 2(y^2 + z^2) & 2(xy - zr) & 2(xz + yr) \\
    2(xy + zr) & 1-2(x^2+z^2) & 2(yz - xr) \\
    2(xz - yr) & 2(yz + xr) & 1 - 2(x^2 + y^2)
  \end{bmatrix}
  $$

* 刚刚求出的误差对旋转矩阵的梯度矩阵：$$U = \frac{\partial Loss}{\partial R} = S \cdot 2 \cdot S \cdot R \cdot \frac{\partial Loss}{\partial \Sigma_{gaussian3d}} $$
* 根据标量对矩阵求导的链式法则（Frobenius 内积），可以分别求出：
  
  $$
    \begin{aligned}
      \frac{\partial Loss}{\partial r} &= \sum_{i=1}^3 \sum_{j=1}^3 \frac{\partial Loss}{\partial R_{ij}} \cdot \frac{\partial R_{ij}}{\partial r} &= \sum_{i=1}^3 \sum_{j=1}^3 U_{ij} \cdot \frac{\partial R_{ij}}{\partial r} \\
      \frac{\partial Loss}{\partial x} &= \sum_{i=1}^3 \sum_{j=1}^3 \frac{\partial Loss}{\partial R_{ij}} \cdot \frac{\partial R_{ij}}{\partial x} &= \sum_{i=1}^3 \sum_{j=1}^3 U_{ij} \cdot \frac{\partial R_{ij}}{\partial x} \\
      \frac{\partial Loss}{\partial y} &= \sum_{i=1}^3 \sum_{j=1}^3 \frac{\partial Loss}{\partial R_{ij}} \cdot \frac{\partial R_{ij}}{\partial y} &= \sum_{i=1}^3 \sum_{j=1}^3 U_{ij} \cdot \frac{\partial R_{ij}}{\partial y} \\
      \frac{\partial Loss}{\partial z} &= \sum_{i=1}^3 \sum_{j=1}^3 \frac{\partial Loss}{\partial R_{ij}} \cdot \frac{\partial R_{ij}}{\partial z} &= \sum_{i=1}^3 \sum_{j=1}^3 U_{ij} \cdot \frac{\partial R_{ij}}{\partial z}
    \end{aligned}
  $$
  
  $$
    \begin{aligned}
      \frac{\partial Loss}{\partial r} &= 2 \cdot (x U_{23} - x U_{32} + y U_{31} - y U_{13} + z U_{12} - z U_{21}) \\
      \frac{\partial Loss}{\partial x} &= 2 \cdot (-2x U_{22} - 2x U_{33} + y U_{12} + y U_{21} + z U_{13} + z U_{31} + r U_{23} - r U_{32}) \\
      \frac{\partial Loss}{\partial y} &= 2 \cdot (x U_{12} + x U_{21} - 2y U_{11} - 2y U_{33} + z U_{23} + z U_{32} + r U_{31} - r U_{13}) \\
      \frac{\partial Loss}{\partial z} &= 2 \cdot (x U_{13} + x U_{31} + y U_{23} + y U_{32} - 2z U_{11} - 2z U_{22} + r U_{12} - r U_{21})
    \end{aligned}
  $$

#### 求解5：3D 均值梯度 $\frac{\partial L}{\partial \mu_{gaussian3d}}$
3D均值(即3D高斯椭球球心)在前向传播中影响了三大模块：2D高斯的均值、颜色、2D高斯的协方差矩阵，所以其最终的梯度应该由三部分构成，形如$\frac{\partial Loss}{\partial \mu_{3D}} = \text{Grad}_{A} (\text{位置}) + \text{Grad}_{B} (\text{颜色}) + \text{Grad}_{C} (\text{形状})$
##### 由$\frac{\partial Loss}{\partial \mu_{gaussian2d}}$求$\frac{\partial Loss}{\partial \mu_{gaussian3d}}(\text{位置})$
* 首先，明确$\mu_{gaussian2d}$v不是在像素坐标系上而是在NDC坐标系上(```renderCUDA```部分已经进行了预处理)，所以现在的流程就是将NDC坐标系的梯度回传给世界坐标系。其次由于涉及到透视变换与可能存在的相机旋转，空间坐标系中任意一个方向的坐标变化($x$ 或 $y$ 或 $z$)都会引发NDC坐标系中$x$和$y$两个方向的变化，所以这里应该使用多元微分学中的**全导数公式**，即：

$$
    \begin{aligned}
        \frac{\partial Loss}{\partial x_{3D}} &= \frac{\partial Loss}{\partial x_{ndc}} \cdot \frac{\partial x_{ndc}}{\partial x_{3D}} + \frac{\partial Loss}{\partial y_{ndc}} \cdot \frac{\partial y_{ndc}}{\partial x_{3D}} \\
        \frac{\partial Loss}{\partial y_{3D}} &= \frac{\partial Loss}{\partial x_{ndc}} \cdot \frac{\partial x_{ndc}}{\partial y_{3D}} + \frac{\partial Loss}{\partial y_{ndc}} \cdot \frac{\partial y_{ndc}}{\partial y_{3D}} \\
        \frac{\partial Loss}{\partial z_{3D}} &= \frac{\partial Loss}{\partial x_{ndc}} \cdot \frac{\partial x_{ndc}}{\partial z_{3D}} + \frac{\partial Loss}{\partial y_{ndc}} \cdot \frac{\partial y_{ndc}}{\partial z_{3D}} 
    \end{aligned}
$$

* 现在需要求$\frac{\partial x_{ndc}}{\partial x_{3D}}$, $\frac{\partial y_{ndc}}{\partial x_{3D}}$, $\frac{\partial x_{ndc}}{\partial y_{3D}}$, $\frac{\partial y_{ndc}}{\partial y_{3D}}$, $\frac{\partial x_{ndc}}{\partial z_{3D}}$, $\frac{\partial y_{ndc}}{\partial z_{3D}}$

$$
    \begin{aligned}
        \frac{\partial x_{ndc}}{\partial x} = \frac{\partial (\frac{p_x}{p_w})}{\partial x}
        = \frac{1}{p_w}\frac{\partial p_x}{\partial x} - \frac{p_x}{p_w^2}\frac{\partial p_w}{\partial x} 
        = \frac{P_{00}}{p_w} - \frac{p_x}{p_w^2} \cdot P_{30} \\
        \frac{\partial y_{ndc}}{\partial x} = \frac{\partial (\frac{p_y}{p_w})}{\partial x}
        = \frac{1}{p_w}\frac{\partial p_y}{\partial x} - \frac{p_y}{p_w^2}\frac{\partial p_w}{\partial x} 
        = \frac{P_{10}}{p_w} - \frac{p_y}{p_w^2} \cdot P_{30} \\
        \frac{\partial x_{ndc}}{\partial y} = \frac{\partial (\frac{p_x}{p_w})}{\partial y}
        = \frac{1}{p_w}\frac{\partial p_x}{\partial y} - \frac{p_x}{p_w^2}\frac{\partial p_w}{\partial y} 
        = \frac{P_{01}}{p_w} - \frac{p_x}{p_w^2} \cdot P_{31} \\
        \frac{\partial y_{ndc}}{\partial y} = \frac{\partial (\frac{p_y}{p_w})}{\partial y}
        = \frac{1}{p_w}\frac{\partial p_y}{\partial y} - \frac{p_y}{p_w^2}\frac{\partial p_w}{\partial y} 
        = \frac{P_{11}}{p_w} - \frac{p_y}{p_w^2} \cdot P_{31} \\
        \frac{\partial x_{ndc}}{\partial z} = \frac{\partial (\frac{p_x}{p_w})}{\partial z}
        = \frac{1}{p_w}\frac{\partial p_x}{\partial z} - \frac{p_x}{p_w^2}\frac{\partial p_w}{\partial z} 
        = \frac{P_{02}}{p_w} - \frac{p_x}{p_w^2} \cdot P_{32} \\
        \frac{\partial y_{ndc}}{\partial z} = \frac{\partial (\frac{p_y}{p_w})}{\partial z}
        = \frac{1}{p_w}\frac{\partial p_y}{\partial z} - \frac{p_y}{p_w^2}\frac{\partial p_w}{\partial z} 
        = \frac{P_{12}}{p_w} - \frac{p_y}{p_w^2} \cdot P_{32} \\
    \end{aligned}
$$

> MP变换：
> 
> $$p_{hom} = P \cdot (x, y, z, 1)^T$$ 
> $$p_x = P_{00}x + P_{01}y + P_{02}z + P_{03}$$  
> $$p_y = P_{10}x + P_{11}y + P_{12}z + P_{13}$$  
> $$p_w = P_{30}x + P_{31}y + P_{32}z + P_{33}$$ 


* 最后全部代入全导数公式即可：

$$
    \begin{aligned}
        \frac{\partial Loss}{\partial \mu_{gaussian3d}}(\text{位置}) = 
        \begin{bmatrix}
            \left( \frac{P_{00}}{p_w} - P_{30}\frac{p_x}{p_w^2} \right) \cdot \frac{\partial Loss}{\partial x_{ndc}} + \left( \frac{P_{10}}{p_w} - P_{30}\frac{p_y}{p_w^2} \right) \cdot \frac{\partial Loss}{\partial y_{ndc}} \\
            \left( \frac{P_{01}}{p_w} - P_{31}\frac{p_x}{p_w^2} \right) \cdot \frac{\partial Loss}{\partial x_{ndc}} + \left( \frac{P_{11}}{p_w} - P_{31}\frac{p_y}{p_w^2} \right) \cdot \frac{\partial Loss}{\partial y_{ndc}} \\
            \left( \frac{P_{02}}{p_w} - P_{32}\frac{p_x}{p_w^2} \right) \cdot \frac{\partial Loss}{\partial x_{ndc}} + \left( \frac{P_{12}}{p_w} - P_{32}\frac{p_y}{p_w^2} \right) \cdot \frac{\partial Loss}{\partial y_{ndc}}
        \end{bmatrix}
    \end{aligned}
$$

##### 由 $\frac{\partial Loss}{\partial RGB_{gaussian2d}}$ 求 $\frac{\partial Loss}{\partial \mu_{gaussian3d}}(\text{颜色})$
> 回顾球谐函数
> 
> $$
    \begin{aligned}
        Color_{final} &= \max(Color_{raw}, 0.0) \\
        Color_{raw} &= C_{l=0} + C_{l=1} + C_{l=2} + C_{l=3} + 0.5 \\
        &= C_0 \cdot sh[0] \\
        &+ -C_1 \cdot y \cdot sh[1] + C_1 \cdot z \cdot sh[2] - C_1 \cdot x \cdot sh[3] \\
        &+ C_{2,0} \cdot (x \cdot y) \cdot sh[4] \\
          &+ C_{2,1} \cdot (y \cdot z) \cdot sh[5] \\
          &+ C_{2,2} \cdot (2z^2 - x^2 - y^2) \cdot sh[6] \\
          &+ C_{2,3} \cdot (x \cdot z) \cdot sh[7] \\
          &+ C_{2,4} \cdot (x^2 - y^2) \cdot sh[8] \\
        &+ C_{3,0} \cdot y(3x^2 - y^2) \cdot sh[9] \\
          &+ C_{3,1} \cdot (xyz) \cdot sh[10] \\
          &+ C_{3,2} \cdot y(4z^2 - x^2 - y^2) \cdot sh[11] \\
          &+ C_{3,3} \cdot z(2z^2 - 3x^2 - 3y^2) \cdot sh[12] \\
          &+ C_{3,4} \cdot x(4z^2 - x^2 - y^2) \cdot sh[13] \\
          &+ C_{3,5} \cdot z(x^2 - y^2) \cdot sh[14] \\
          &+ C_{3,6} \cdot x(x^2 - 3y^2) \cdot sh[15]
    \end{aligned}
> $$
* 由于这里3D均值通过影响归一化视角方向$\text{dir}$从而间接影响了RGB颜色，所以我们应该先利用**全导数法则**求$\frac{\partial Loss}{\partial \text{dir}}$:
    1. 首先明确归一化视角：
    
    $$
        \begin{aligned}
            dir &= (dir_x, dir_y, dir_z) = \left( \frac{x'}{d}, \frac{y'}{d}, \frac{z'}{d} \right) \\
            &= \left( \frac{x'}{\sqrt{(x')^2 + (y')^2 + (z')^2}}, \frac{y'}{\sqrt{(x')^2 + (y')^2 + (z')^2}}, \frac{z'}{\sqrt{(x')^2 + (y')^2 + (z')^2}} \right) \\
            &=  \left(\frac{x - c_x}{\sqrt{(x - c_x)^2 + (y - c_y)^2 + (z - c_z)^2}}, \frac{y - c_y}{\sqrt{(x - c_x)^2 + (y - c_y)^2 + (z - c_z)^2}}, \frac{z - c_z}{\sqrt{(x - c_x)^2 + (y - c_y)^2 + (z - c_z)^2}} \right)
        \end{aligned}
    $$
    2. 分别展开求$\frac{\partial Loss}{\partial dir_x}$、$\frac{\partial Loss}{\partial dir_y}$、$\frac{\partial Loss}{\partial dir_z}$：
    
    $$
        \begin{aligned}
            \frac{\partial Loss}{\partial x_{dir}} &= \frac{\partial Loss}{\partial Color_{RGB}} \cdot \frac{\partial Color_{RGB}}{\partial x_{dir}} \\
            &= \sum_{channel \in \{R,G,B\}} \frac{\partial Loss}{\partial Color_{channel}} \cdot \frac{\partial Color_{raw}}{\partial x} \\
            \frac{\partial Loss}{\partial y_{dir}} &= \frac{\partial Loss}{\partial Color_{RGB}} \cdot \frac{\partial Color_{RGB}}{\partial y_{dir}} \\
            &= \sum_{channel \in \{R,G,B\}} \frac{\partial Loss}{\partial Color_{channel}} \cdot \frac{\partial Color_{raw}}{\partial y} \\
            \frac{\partial Loss}{\partial z_{dir}} &= \frac{\partial Loss}{\partial Color_{RGB}} \cdot \frac{\partial Color_{RGB}}{\partial z_{dir}} \\
            &= \sum_{channel \in \{R,G,B\}} \frac{\partial Loss}{\partial Color_{channel}} \cdot \frac{\partial Color_{raw}}{\partial z}
        \end{aligned}
    $$

    3. 合并得到$\frac{\partial Loss}{\partial \text{dir}}$：
    
    $$
    \frac{\partial Loss}{\partial \text{dir}} = 
    \begin{bmatrix}
        \frac{\partial Loss}{\partial x_{dir}} \\
        \frac{\partial Loss}{\partial y_{dir}} \\
        \frac{\partial Loss}{\partial z_{dir}}
    \end{bmatrix}
    $$

* 下一步求$\frac{\partial dir}{\partial \mu_{gaussian3d}}$ ，由于相机坐标 $cam_{pos} = (c_x, c_y, c_z)$ 是一个常数，所以对$\mu_{gaussian3d}$就等于对$(x', y', z') = (x - c_x, y - c_y, z - c_z)$求导：
    
    $$
        \begin{aligned}
            \Delta &= \mu_{gaussian3d} - cam\_pos =  (x - c_x, y - c_y, z - c_z) = (x', y', z') \\
            \because dir &= \frac{\Delta}{d}  = \frac{\Delta}{\sqrt{(x')^2 + (y')^2 + (z')^2}} \\
            \frac{\partial dir}{\partial \mu_{gaussian3d}} = \frac{\partial dir}{\partial \Delta}
                &= \begin{bmatrix}
                    \frac{\partial dir_x}{\partial x'} & \frac{\partial dir_x}{\partial y'} & \frac{\partial dir_x}{\partial z'} \\
                    \frac{\partial dir_y}{\partial x'} & \frac{\partial dir_y}{\partial y'} & \frac{\partial dir_y}{\partial z'} \\
                    \frac{\partial dir_z}{\partial x'} & \frac{\partial dir_z}{\partial y'} & \frac{\partial dir_z}{\partial z'}
                \end{bmatrix}
                = \begin{bmatrix}
                \frac{1}{d} - \frac{(x')^2}{d^3} & -\frac{x' y'}{d^3} & -\frac{x' z'}{d^3} \\
                -\frac{x' y'}{d^3} & \frac{1}{d} - \frac{(y')^2}{d^3} & -\frac{y' z'}{d^3} \\
                -\frac{x' z'}{d^3} & -\frac{y' z'}{d^3} & \frac{1}{d} - \frac{(z')^2}{d^3}
                \end{bmatrix} \\
                &= \frac{1}{d} \left( I - \frac{\Delta \cdot \Delta^T}{d^2} \right) \\
                &= \frac{1}{d} \left( I - dir \cdot dir^T \right)
        \end{aligned}
    $$

* 最后利用矩阵的链式求导法则，得到:
  
  $$
    \begin{aligned}
        \frac{\partial Loss}{\partial \mu_{gaussian3d}}(\text{颜色})
        &= (\frac{\partial dir}{\partial \mu_{gaussian3d}})^T \cdot \frac{\partial Loss}{\partial \text{dir}} \\
        &= \frac{1}{d} \left( I - dir \cdot dir^T \right) \cdot \frac{\partial Loss}{\partial \text{dir}} \\
        &= \frac{1}{d} \left( v_{grad} - dir \cdot (dir^T \cdot \frac{\partial Loss}{\partial \text{dir}}) \right)
    \end{aligned}
  $$

##### 由$\frac{\partial Loss}{\partial \Sigma^{-1}_{gaussian2d}}$求$\frac{\partial Loss}{\partial \mu_{gaussian3d}}(\text{形状})$
* 由于3D均值藏在 $J$ 矩阵中，所以需要根据 EWA Splatting 过程首先计算 $\frac{\partial Loss}{\partial J}$:
  
  $$
    \begin{aligned}
        \because \Sigma_{2D} &= J W \Sigma_{3D} W^T J^T \\
        \because t_{cam} &= (t_x, t_y, t_z) = W \cdot \mu_{gaussian3d} + t_{cam} \\
        \text{设相机坐标系下的 3D 协方差:} \space
        V &= W \Sigma_{3D} W^T \\
        \Sigma_{2D} &= J \cdot V \cdot J^T = 
        \begin{bmatrix}
        \frac{f_x}{t_z} & 0 & -f_x \frac{t_x}{t_z^2} \\ 
        0 & \frac{f_y}{t_z} & -f_y \frac{t_y}{t_z^2} \\ 
        0 & 0 & 0
        \end{bmatrix} \cdot V \cdot 
        \begin{bmatrix}
        \frac{f_x}{t_z} & 0 & -f_x \frac{t_x}{t_z^2} \\ 
        0 & \frac{f_y}{t_z} & -f_y \frac{t_y}{t_z^2} \\ 
        0 & 0 & 0
        \end{bmatrix}^T \\
        &= \begin{bmatrix}
        \frac{f_x}{t_z} & 0 & -f_x \frac{t_x}{t_z^2} \\ 
        0 & \frac{f_y}{t_z} & -f_y \frac{t_y}{t_z^2}
        \end{bmatrix} \cdot V \cdot 
        \begin{bmatrix}
        \frac{f_x}{t_z} & 0 & -f_x \frac{t_x}{t_z^2} \\ 
        0 & \frac{f_y}{t_z} & -f_y \frac{t_y}{t_z^2}
        \end{bmatrix}^T \\
        d\Sigma_{2D} &= d(J \cdot V) \cdot J^T + J \cdot V \cdot dJ^T \\
        代入d(Loss) &= \text{tr}(\frac{\partial Loss}{\partial \Sigma_{gaussian2d}} \cdot d\Sigma_{gaussian2d}) = \text{tr}(\frac{\partial Loss}{\partial \Sigma_{gaussian2d}} \cdot (d(J \cdot V) \cdot J^T + J \cdot V \cdot dJ^T)) \\
        &= 2 \text{tr}(V \cdot J^T \cdot \frac{\partial Loss}{\partial \Sigma_{gaussian2d}} \cdot dJ) \\
        \therefore \frac{\partial Loss}{\partial J} &= 2 \cdot \frac{\partial Loss}{\partial \Sigma_{gaussian2d}} \cdot J \cdot V
    \end{aligned}
  $$

> 这里用到了矩阵求导第一性原理：
> 
> $$d(Loss) = \text{tr}\left( \left(\frac{\partial Loss}{\partial X}\right)^T dX \right)$$
* 从$\frac{\partial Loss}{\partial J}$ 回退到 $\frac{\partial Loss}{\partial (t_x, t_y, t_z)}$(相机坐标系下的3D均值)：
  
  $$
    \begin{aligned}
        令G_J &= \frac{\partial Loss}{\partial J} = 2 \cdot \frac{\partial Loss}{\partial \Sigma_{gaussian2d}} \cdot J \cdot V \\
        \frac{\partial J}{\partial t_x} &=  
        \begin{bmatrix}
            0 & 0 & -\frac{f_x}{t_z^2} \\
            0 & 0 & 0 \\
            0 & 0 & 0
        \end{bmatrix} = \frac{\partial J_{02}}{\partial t_x}\\
        \frac{\partial J}{\partial t_y} &=  
        \begin{bmatrix}
            0 & 0 & 0 \\
            0 & 0 & -\frac{f_y}{t_z^2} \\
            0 & 0 & 0
        \end{bmatrix} = \frac{\partial J_{12}}{\partial t_x}\\
        \frac{\partial J}{\partial t_z} &=  
        \begin{bmatrix}
            -\frac{f_x}{t_z^2} & 0 & 2\frac{f_x \cdot t_x}{t_z^3} \\
            0 & -\frac{f_y}{t_z^2} & 2\frac{f_y \cdot t_y}{t_z^3} \\
            0 & 0 & 0
        \end{bmatrix} \\
        \therefore \frac{\partial Loss}{\partial t_x} &= G_{J}^{02} \cdot \frac{\partial J_{02}}{\partial t_x} = G_{J}^{02} \cdot \left( -\frac{f_x}{t_z^2} \right) \\
        \frac{\partial Loss}{\partial t_y} &= G_{J}^{12} \cdot \frac{\partial J_{12}}{\partial t_y} = G_{J}^{12} \cdot \left( -\frac{f_y}{t_z^2} \right) \\
        \frac{\partial Loss}{\partial t_z} &= G_{J}^{00} \frac{\partial J_{00}}{\partial t_z} + G_{J}^{11} \frac{\partial J_{11}}{\partial t_z} + G_{J}^{02} \frac{\partial J_{02}}{\partial t_z} + G_{J}^{12} \frac{\partial J_{12}}{\partial t_z} \\
        &= G_{J}^{00}\left(-\frac{f_x}{t_z^2}\right) + G_{J}^{11}\left(-\frac{f_y}{t_z^2}\right) + G_{J}^{02}\left(\frac{2 f_x t_x}{t_z^3}\right) + G_{J}^{12}\left(\frac{2 f_y t_y}{t_z^3}\right)
    \end{aligned}
  $$
* 由相机坐标系下的3D均值的梯度$\frac{\partial Loss}{\partial (t_x, t_y, t_z)}$回到世界坐标系下的均值的梯度 $\frac{\partial Loss}{\partial \mu_{gaussian3d}}(\text{形状})$
  
  $$
    \begin{aligned}
        &\because t_{cam} = (t_x, t_y, t_z) = W \cdot \mu_{gaussian3d} + t_{cam} \\
        &\therefore \frac{\partial Loss}{\partial \mu_{gaussian3d}}(\text{形状}) = W^T \cdot \frac{\partial Loss}{\partial (t_x, t_y, t_z)}
    \end{aligned}
  $$
> 经历了千辛万苦推到了这里，然而```CUDA```代码中并没有考虑这一部分，可能是为了节省算力考虑吧 $\dots$