## Gaussian Splatting 面临的个问题:
* 弱纹理
* reflective surface
* gaussian splatting如果input是由不同camera照的有光线variation的话render出来appearance很不uniform
* 对于两张光照不同的photos的融合,使用的是添加一个black gaussian sphere
---
## gaussian splatting forward 的近似步骤:
* 计算投影出来的圆(近似)的半径 - preprocessCUDA
* 计算每个圆覆盖了哪些像素:    - preprocessCUDA
  通过将screen分为一个个 $16 \times 16 pixels $ 的小块,近似确定每个圆覆盖了哪些区域
* 计算每个高斯的前后顺序
* 计算每个像素的颜色           - renderCUDA
---
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
### Rendering 部分的梯度计算 —— 链式法则
#### 前置准备工作
* 已知 : $\frac{\partial Loss}{\partial {RGB_{pixel}}} $
  | 求解 |
  |-|
  | $\frac{\partial Loss}{\partial RGB_{gaussian2d}} $ |
  | $\frac{\partial Loss}{\partial opacity_{gaussian2d}} $ |
  | $\frac{\partial Loss}{\partial \Sigma^{-1}_{gaussian2d}} $ |
  | $\frac{\partial Loss}{\partial \mu_{gaussian2d}}$ |
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
---
## 每一个 pixel 处颜色的获取流程

### 第一步：计算像素到高斯中心的“马氏距离” (Mahalanobis Distance)

假设我们现在正在计算屏幕上坐标为 $(x, y)$ 的像素。根据排序，我们知道有一个 2D 高斯椭圆覆盖了这个像素。这个高斯的中心点在 $(x_0, y_0)$。像素到中心的偏移量为：
$$
\begin{equation}
\begin{aligned}
  dx &= x - x_0 \\
  dy &= y - y_0
\end{aligned}
\end{equation}
$$
此时，我们前面辛辛苦苦算出来的逆协方差矩阵 conic (包含 $a, b, c$ 三个值) 终于派上用场了！
代码会用它们算出一个“距离指数 (Power)”：
$$
\begin{equation}
  \begin{aligned}
    Power &= -\frac{1}{2} (X - \mu)^T \Sigma^{-1} (X - \mu) \\
    &= -0.5 \times (a \cdot dx^2 + c \cdot dy^2 + 2 \cdot b \cdot dx \cdot dy) \\
  \end{aligned}
\end{equation}
$$

  > (注：这里 $a, b, c$ 对应代码里的 conic.z, conic.y, conic.x)

这就是二维高斯分布公式的核心！ 它不仅仅是用欧氏距离画个圆，而是用协方差描述的椭圆形状来衡量“**这个像素偏离中心有多远”**。

### 第二步：计算高斯衰减权重 (Gaussian Weight)
算出指数后，直接求自然底数 $e$ 的幂：
$$Weight = \exp(Power)$$
  > 物理意义： 如果像素刚好在高斯中心（$dx=0, dy=0$），那么 $Power = 0$，$\exp(0) = 1$。权重最高。像素越靠近椭圆的边缘，$Power$ 是个越来越小的负数，$Weight$ 就会呈钟形曲线迅速衰减，接近于 0。

### 第三步：计算该像素感受到的真实不透明度 (Alpha)
高斯体本身有一个基础的不透明度（假设叫 Base_Opacity，它是网络训练出来的属性）。当前这个像素被高斯体覆盖后，真正感受到的不透明度 $\alpha$ 是：
$$\alpha = Base\_Opacity \times Weight$$
  > (注：在这里，如果算出来的 $\alpha$ 小于某个极小值比如 $1/255$，代码会直接跳过这个高斯不渲染它，以节省性能。)

### 第四步：颜色的累加（Alpha Blending / 体渲染积分）
这是最关键的一步。一个像素通常不是被一个高斯盖住的，而是被几十上百个高斯层层叠叠盖住的！在 renderCUDA 之前，程序已经利用 depth（深度）把覆盖该像素的所有高斯从前向后 (Front-to-Back) 排好了序。
我们假设像素初始没有颜色（黑色 $C=0$），并且初始的透射率（光能穿透的比例） $T=1.0$。然后我们顺着视线，从离眼睛最近的第一个高斯开始，逐个往下算：
**对于第 $i$ 个高斯：**
1. 取出它在这个视角的颜色 $c_i$（通过球谐函数算出来的 RGB）。
2. 算出它在这个像素上的真实不透明度 $\alpha_i$（上面第三步算的值）。
3. 把颜色加到像素上：$C = C + T \times \alpha_i \times c_i$
4. 更新剩余的透射率：$T = T \times (1 - \alpha_i)$
---
## 3DGS优于传统光栅化渲染与NeRF的地方
### 1. 从“硬边界”到“软过渡” (Soft Splatting)
* 在传统的游戏渲染里，模型是由无数个三角形面片组成的。一个像素要么在一个三角形内部（权重为 1，完全遮挡后面），要么在外部（权重为 0）。
  > 缺点:这种“非黑即白”的硬边界导致它很难被深度学习网络去优化（因为不可微）。
* 而 3DGS 用了二维高斯分布。像素点 $(x, y)$ 受到高斯球的影响是一个 $0 \sim 1$ 之间连续变化的平滑曲线。
  >这种“软过渡”不仅让边缘融合得极其自然，更关键的是——它是处处可导的！这就是为什么后来在 backward.cu（反向传播）中，网络能根据最终颜色的误差，一点点去微调这些高斯球的位置、大小和不透明度。
### 2. 球谐函数 (SH) 的独立视角解耦
* **颜色的权重**是由高斯分布（形状）决定的;
* **颜色本身**是由球谐函数决定的;
* **协方差**决定了“这个像素能不能看到我，我看上去有多大”;
* **球谐函数**决定了“你从哪个特定的角度在看我，我该反射出什么颜色”。
 > 这两个属性完美解耦，让 3DGS 既能表现出毛玻璃、水面那种随视角变化的强烈高光（SH 发力），又能保持物理形状的稳定（高斯椭球发力）。

### 3. 体渲染 (Volume Rendering) 的极速降维打击
* **光线透射率衰减 (Transmittance Attenuation)**:
  以前的 NeRF 为了算这一点，必须顺着光线在三维空间里一步步“盲人摸象”般地采样，计算量巨大，渲染极其缓慢。
  而 3DGS 直接把所有高斯“拍扁”在屏幕上（Splatting），然后通过深度的前缀排序，变成了纯粹的 2D 图像叠加合成 (Alpha Blending)。它用最廉价的 2D 像素层叠运算，完美伪装出了 3D 空间的物理遮挡和光线穿透效果，这就是它能跑到实时 100 帧以上的根本原因！
---

## 回顾图形学中的 MVP 变换
### 1. View Transform 视图变换 (viewmatrix)
* **把世界“搬”到相机的面前：** 它负责将高斯球的坐标从世界坐标系 (World Space) 转换到相机坐标系 (View Space)。在相机坐标系中，相机的镜头永远是坐标原点 $(0,0,0)$，相机的正前方通常是 Z 轴。
* 它是一个 $4 \times 4$ 的仿射变换矩阵，本质上包含了相机的旋转 (Rotation, $3 \times 3$) 和平移 (Translation, $3 \times 1$)：
  $$V = \begin{bmatrix} R_{00} & R_{01} & R_{02} & t_x \\ R_{10} & R_{11} & R_{12} & t_y \\ R_{20} & R_{21} & R_{22} & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
  > 特点 1（刚体变换）： 它只做旋转和平移，绝对不会改变高斯球的大小和形状。
  > 特点 2（底层全 01）： 它的最后一行永远是 $[0, 0, 0, 1]$。

### 2. Project Transform 投影变换 (projmatrix 源码中对应的是全局投影矩阵)
* **产生“近大远小”的透视扭曲，并准备降维：** 在 3DGS 的这个 CUDA kernel 中，由于 projmatrix 实际上等于 $P \times V$，所以它是一个“一步到位”的矩阵：直接把高斯球从世界坐标系 (World Space) 狠狠地压缩进齐次裁剪空间 (Homogeneous Clip Space)，即获得NDC坐标(归一化设备坐标)。
* 透视投影矩阵的推导非常复杂（受视场角 FOV、屏幕宽高比、近裁剪面、远裁剪面影响），它也是一个 $4 \times 4$ 的矩阵，但它有着极其特殊的最后一行：
  $$M_{proj} = P \times V = \begin{bmatrix} m_{00} & m_{01} & m_{02} & m_{03} \\ m_{10} & m_{11} & m_{12} & m_{13} \\ m_{20} & m_{21} & m_{22} & m_{23} \\ m_{30} & m_{31} & m_{32} & m_{33} \end{bmatrix}$$
  > 特点 1（非线性变形）： 经过这个矩阵变换后，原本平行的线可能会相交（这就是透视效果）。在这个空间里，高斯体的形状已经被“拉扯”变形了。
  > 特点 2（最后一行不再是 0001）： 这是它和 viewmatrix 最大的区别！它的最后一行通常与 $Z$ 坐标相关（例如 $[0, 0, 1, 0]$ 或类似形式）。这意味着，世界坐标 $(X,Y,Z,1)$ 乘上它之后，得到的四维结果 $(x, y, z, w)$ 中，$w$ 不再是 1，而是包含了该点的深度信息！ 
---
## 为什么不让神经网络直接学习和预测 三维高斯分布的协方差矩阵 $\Sigma$
* **背景：** 在数学上，代表高斯分布形状的协方差矩阵 $\Sigma$ 必须是半正定矩阵 (Positive Semi-Definite)。
  $$
  \Sigma = \begin{bmatrix}
    a & b & c \\ b & d & e \\ c & e & f
  \end{bmatrix}
  $$
* **问题：** 如果让神经网络自由地去预测这 6 个数字，在梯度下降的反向传播过程中，稍微更新一下，这个矩阵可能就不再是半正定的了。一旦变成非半正定矩阵，高斯分布在物理上就“碎掉”了（体积变成负数，或者变成复数），程序直接崩溃（出 NaN）。
* **3DGS的解决方案：** 任何半正定矩阵 $\Sigma$ 都可以被分解为一个旋转矩阵 $R$ 和一个对角缩放矩阵 $S$ 的组合。只要我们让网络去预测独立的缩放 $S$ 和旋转 $R$，然后通过数学公式把它们拼起来，构造出来的 $\Sigma$ 永远百分之百是合法的半正定矩阵！ 这在深度学习中被称为**“协方差的参数化重构”**。
* 论文中的协方差参数重构流程：
  $$\Sigma = R S S^T R^T$$
  > 注：代码中为了适应列主序等底层逻辑，写成了 $\Sigma = M^T M$ (其中 $M=SR$)，在数学等效性和网络优化上殊途同归。
---
## 重新理解 Jacobi 矩阵 与 EWA Splat操作
协方差矩阵代表的是一个椭球形状，这是一个线性变换概念。但是，透视投影是一个非线性的变换（近大远小，除以 $z$ 的操作是曲线的）。线性矩阵无法直接经过非线性变换！因此，我们需要用到微积分中的泰勒展开：在局部极其微小的区域内，用一条直线来近似曲线。 雅可比矩阵就是透视投影在这个高斯体中心点处的局部一阶线性近似。 
> **数学推导：** 在相机空间中，一个点 $(x_c, y_c, z_c)$ 投影到 2D 屏幕坐标 $(u, v)$ 的公式为：$$u = f_x \frac{x_c}{z_c}$$$$v = f_y \frac{y_c}{z_c}$$雅可比矩阵 $J$ 是由这组方程对 $x_c, y_c, z_c$ 的偏导数组成的 $3 \times 3$ 矩阵：
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

所以
$$
\Sigma' = J W \Sigma W^T J^T
$$
<!-- $$
\begin{equation}
  \begin{aligned}
    \Sigma' &= J W \Sigma W^T J^T \\
    % &= J W \Sigma W^T J^T\\
    % &= 
    % \begin{bmatrix}
    % \frac{f_x}{z_c} & 0 & -f_x \frac{x_c}{z_c^2} \\ 
    % 0 & \frac{f_y}{z_c} & -f_y \frac{y_c}{z_c^2} \\ 
    % 0 & 0 & 0 
    % \end{bmatrix} \cdot{
    %   \begin{bmatrix}
    %   r_{00} & r_{01} & r_{02} \\ 
    %   r_{10} & r_{11} & 1_{12} \\ 
    %   r_{20} & r_{21} & r_{22} 
    %   \end{bmatrix}
    % }
    % \cdot
    % \Sigma
    % \cdot
    %   \begin{bmatrix}
    %   r_{00} & r_{01} & r_{02} \\ 
    %   r_{10} & r_{11} & 1_{12} \\ 
    %   r_{20} & r_{21} & r_{22} 
    %   \end{bmatrix}^T
    % \cdot
    % \begin{bmatrix}
    % \frac{f_x}{z_c} & 0 & -f_x \frac{x_c}{z_c^2} \\ 
    % 0 & \frac{f_y}{z_c} & -f_y \frac{y_c}{z_c^2} \\ 
    % 0 & 0 & 0 
    % \end{bmatrix}^T
  \end{aligned}
\end{equation}
$$ -->
---
## 再谈 球谐函数（Spherical Harmonics, 简称 SH） 
* 角色 ：用来表示高斯椭球“视角依赖颜色（View-dependent Color）”的核心数学工具。
  > 现实世界中的物体在不同角度下看，颜色和亮度是会变化的（比如金属的反光、光碟的变色）。3DGS 没有使用复杂的物理光照模型，而是用一组纯数学的正交基函数（也就是 SH）来拟合这种随着视角变化的颜色。
* 核心公式：
  在 3DGS 中，给定一个观察方向向量 $v = (x, y, z)$（通常是归一化的单位向量），计算某个高斯点最终 RGB 颜色 $C(v)$ 的总公式为：
  $$ C(v) = \sum_{l=0}^{L_{max}} \sum_{m=-l}^{l} c_{l}^m Y_l^m(v) $$
  > * $C(v)$: 最终输出的颜色。它是一个三维向量 $(R, G, B)$。
  > * $l$: 球谐函数的阶数（Degree）。3DGS 官方实现中，默认最大阶数 $L_{max} = 3$。阶数越高，能表达的高频细节（如锐利的镜面高光）就越丰富。
  > * $m$: 球谐函数的序数（Order），取值范围是 $[-l, l]$。对于每一阶 $l$，有 $2l + 1$ 个序数。
  > * $Y_l^m(v)$: 球谐基函数。这是一组固定的数学公式，只与当前的视角方向 $v$ 有关。你可以把它理解为“调色盘上的基础颜料”。
  > * **$c_l^m$: 球谐系数（SH Coefficients）。** 这是 3DGS 在训练过程中**真正需要学习和优化的参数**。由于我们需要表示 RGB，所以每一个 $c_l^m$ 实际上包含三个浮点数 $(c_R, c_G, c_B)$。你可以把它理解为“每种颜料使用的比例”。
* 关于球谐基函数$Y_l^m(v)$：
  为了在 GPU 上高效计算，3DGS 不会使用复杂的球面坐标（$\theta, \phi$），而是直接将 $v$ 映射到笛卡尔坐标系的 $(x, y, z)$ 上。我们将前两阶（$L=0$ 和 $L=1$）展开来看：
  **1. L = 0（第 0 阶：DC 项 / 基础颜色）**
    第 0 阶只有一个基函数（$m=0$），被称为 DC（Direct Current）项。
    $$ Y_0^0 = \frac{1}{2}\sqrt{\frac{1}{\pi}} \approx 0.282095 $$
    > 你会发现 $Y_0^0$ 是一个常数，完全不包含 $x, y, z$。这意味着无论你从哪个角度看，这一项提供的值都是一样的。它代表了高斯点的基础漫反射颜色（Base Color）。

  **2. L = 1（第 1 阶：一次项 / 基础视角依赖）**
    第 1 阶有三个基函数（$m = -1, 0, 1$），它们开始引入对坐标轴的线性依赖。$$ Y_1^{-1} = -\sqrt{\frac{3}{4\pi}} \cdot y \approx -0.488603 \cdot y $$$$ Y_1^0 = \sqrt{\frac{3}{4\pi}} \cdot z \approx 0.488603 \cdot z $$$$ Y_1^1 = -\sqrt{\frac{3}{4\pi}} \cdot x \approx -0.488603 \cdot x $$
    > 当视角方向的 $x, y, z$ 发生变化时，这三项的值会线性变化，从而在不同方向上叠加不同的颜色，产生平滑的渐变效果。
  
  **3. 更高阶（L = 2, 3）**
  到了 $L=2$，基函数会包含二次项（如 $xy, yz, 3z^2-1$ 等）；到了 $L=3$，会包含三次项。它们负责拟合非常复杂的视角反射。
* 最终计算一个点的通道颜色
  假设我们要计算一个高斯点在视角 $v = (x,y,z)$ 下的红色（R）通道值，且只使用前两阶（$L_{max}=1$），具体的计算代码逻辑就是这样的累加：
  $$ R = c_{0,R}^0 \cdot (0.282095) + c_{1,R}^{-1} \cdot (-0.488603 \cdot y) + c_{1,R}^0 \cdot (0.488603 \cdot z) + c_{1,R}^1 \cdot (-0.488603 \cdot x) $$
  > 计算出 $R, G, B$ 后，通常还会通过一个 Sigmoid 函数或简单的裁剪（Clamp），将最终的颜色值限制在 $[0, 1]$ 的范围内，以便在屏幕上渲染。
---
## render过程中使用的 共享内存```__shared__```操作
### 1. ```__shared__```到底是什么 —— 显存 vs 共享内存
在 GPU 的硬件架构中，内存是分等级的：
* **全局内存 (Global Memory)：** 也就是我们常说的“显存”（比如 24GB 的 RTX 4090 显存）。它的容量极大，所有高斯的数据都存在这里，但是它的读取速度非常慢，延迟极高。
* **共享内存 (Shared Memory)：** 这是直接镶嵌在 GPU 计算核心（SM）内部的一块极小、但速度极快（接近寄存器速度）的高速缓存。
* ```__shared__```关键字的作用： 只要变量前面加了它，这个变量就会被分配在共享内存中。它的最重要特性是：**同一个 Block（线程块）内的所有线程**（即我们这个 Tile 里的 256 个像素线程）**，可以共同读写这块内存，并且互相可见**。

### 2. 为什么要在这里使用```__shared__```数组？
* 这是为了解决一个极其严重的“**内存读取拥堵**”问题：
  假设当前 Tile 被 500 个高斯覆盖。如果让 Tile 里的 256 个像素线程各自去全局内存里读取这 500 个高斯的数据，那么 256 个人都要排队去慢速仓库拿同样的东西，显存带宽瞬间就会被撑爆。
* 3DGS 的极其聪明的做法:
  1. BLOCK_SIZE (256) 个线程组成一个“搬砖小队”。
  2. **第一轮 (Round 0)：** 256 个线程每人去全局内存里只拿 1 个高斯的数据，分别放进 collected_id、collected_xy 和 collected_conic_opacity 这三个共享内存的“公共工作台”里。
  3. 此时，公共工作台上整整齐齐地摆放着这 256 个高斯的即食数据。
  4. 接着，256 个像素线程开始飞速地从公共工作台上读取这 256 个高斯的数据，计算它们对自己像素的颜色影响。因为是在芯片内部的高速缓存里读，速度快如闪电！
  5. 算完之后，清空工作台，进入**第二轮 (Round 1)**，再去搬接下来的一批高斯...
* 这就是为什么叫 collected_...（收集来的数据）的原因！这是一种被称为 集体获取 (Collective Fetch / Cooperative Fetch) 的经典 CUDA 优化设计。
---
## 反向传播逻辑顺序梳理 —— 链式求导过程
### 反向传播要更新的参数：
| 三维位置 | 缩放尺度 | 旋转四元数 | 球谐系数 | 基础不透明度 |
  |-|-|-|-|-|
  | $\mu$ | $S$ | $q$ | $SH$ | $\alpha$ |
### 前向传播的pipline顺序很重要：
#### 第一阶段：3D 到 2D 的几何与色彩预处理 (preprocessCUDA)
* **Step 1.1：视锥剔除与深度计算 (Frustum Culling & Depth)**
  **操作：** 将 3D 中心点 means3D 乘以视角矩阵 view_matrix，得到相机坐标系下的位置。
  **判断：** 如果该点的 Z 值不在相机的近平面和远平面之间，或者超出了视锥的扩展范围，直接标记为废弃（radii = 0），退出后续计算。
 **输出记录：** 记录该高斯的深度 depth（用于后续排序）。
* **Step 1.2：构建 3D 协方差矩阵 (Build 3D Covariance)**
  **输入：** 缩放向量 scales (3D)，旋转四元数 rotations (4D)。
  **操作：** 将 rotations 转为旋转矩阵 $R$。将 scales 转为对角缩放矩阵 $S$。
  **计算 3D 协方差：** $$\Sigma = R S S^T R^T$$
  **输出记录：** 3D 协方差矩阵 cov3D（通常包含 6 个独立元素）。
* **Step 1.3：投影计算 2D 协方差 (EWA Splatting)**
  **输入：** cov3D，视角矩阵 $W$，透视投影的雅可比矩阵 $J$。
  **操作：** 计算 2D 协方差 $$\Sigma' = J W \Sigma W^T J^T$$
    > 为了保证数值稳定性，通常会对 $\Sigma'$ 加上一个微小的低通滤波器（对角线加 0.3）。

  **输出求逆：** 对 $\Sigma'$ 求逆，得到代码中的圆锥体参数 conic（包含逆矩阵的 3 个独立元素：左上、右下、右上，用于后续计算指数衰减）。
* **Step 1.4：计算 2D 屏幕坐标 (Project Center)**
  **操作：** 将 3D 中心点 means3D 通过完整投影矩阵（View + Projection）变换到 NDC 坐标系，再缩放到实际的像素网格坐标系。
  **输出记录：** 2D 像素坐标 means2D。
* **Step 1.5：计算视点相关颜色 (Compute Color from SH)**
  **输入：**球谐系数 shs，相机位置 cam_pos，3D 中心 means3D。
  **操作：** 计算视角方向向量 $d = \text{normalize}(\text{means3D} - \text{CameraPosition})$。将 $d$ 代入球谐函数，与 shs 结合计算出当前的 RGB 颜色。
  **输出记录：** 基础颜色 rgb。
* **Step 1.6：计算包围盒与覆盖 Tiles (Bounding Box)**
  **操作：** 根据 2D 协方差计算特征值，求得 2D 椭圆的最大半径（使用 3-sigma 原则，覆盖 99% 的能量）。根据 means2D 和最大半径，计算出这个高斯球在屏幕上覆盖了哪些 16x16 的像素块（Tiles）。
  **输出记录：** 屏幕覆盖半径 radii，以及该高斯覆盖的 Tile 数量 touched。
#### 第二阶段：排序与空间索引组织 (Sorting & Indexing)

---
## 项目调用栈
* Python 层 (train.py): loss = render(viewpoint_camera, gaussians)["render"] (跑在 CPU)
* PyTorch 绑定层 (ext.cpp 等): 将 Python 张量转换为 C++ 数据结构。
* 封装层 (rasterize_points.cu): 提供向外暴露的接口 RasterizeGaussiansCUDA。
* 总指挥 (rasterizer_impl.cu): 执行 CudaRasterizer::Rasterizer::forward，排兵布阵，调用基数排序。（也就是我们即将带你精读的文件）
* 具体打工人 (forward.cu): GPU 上的成千上万个线程，分别执行 preprocessCUDA (投影) 和 renderCUDA (alpha blend 混合渲染)。

---
## AoS 结构体数组 与 SoA 数组结构体
### CPU通常的存储方式为 AoS(Array of Structures, 结构体数组)
假设我们有 $P$ 个高斯体，每个都有 2D 坐标 (means2D) 和深度 (depth)
```cpp
// 定义单个高斯
struct Gaussian2D {
    float2 means2D;
    float depth;
};
// 申请 P 个高斯的数组
Gaussian2D all_gaussians[P];
```
> 这在 CPU 上很完美（面向对象，逻辑清晰）。但是！如果把这个放到 GPU 上，当相邻的线程同时去读取 means2D 时，它们在物理内存上是不连续的（中间隔着 depth）。这会破坏 GPU 最喜欢的内存合并访问（Memory Coalescing），导致读取速度暴跌。

### GPU通常的存储方式为 SoA(Structure of Arrays, 数组结构体)
```cpp
// 这就是 GeometryState 的本质
struct GeometryState {
    float2* means2D; // 指针，指向长度为 P 的数组
    float* depths;   // 指针，指向长度为 P 的数组
    int* radii;      // 指针，指向长度为 P 的数组
    // ... 其他属性
};
```
> 并且 GeometryState 这个对象本身非常小,里面不存具体的数据,全是指针.
---
## 区分两种图像存储方式
| HWC 格式 | CHW格式 |
|-|-|
| 交错存储 | Planar（平面）存储|
| RGB RGB RGB ... | RRR...GGG...BBB... |
| OpenCV | PyTorch / CUDA |