#### 求解5：3D 均值梯度 $\frac{\partial L}{\partial \mu_{gaussian3d}}$
3D均值(即3D高斯椭球球心)在前向传播中影响了三大模块：2D高斯的均值、颜色、2D高斯的协方差矩阵，所以其最终的梯度应该由三部分构成，形如$\frac{\partial Loss}{\partial \mu_{3D}} = \text{Grad}_{A} (\text{位置}) + \text{Grad}_{B} (\text{颜色}) + \text{Grad}_{C} (\text{形状})$
##### 由$\frac{\partial Loss}{\partial \mu_{gaussian2d}}$求$\frac{\partial Loss}{\partial \mu_{gaussian3d}}(\text{位置})$
* 首先，明确$\mu_{gaussian2d}$v不是在像素坐标系上而是在NDC坐标系上(```renderCUDA```部分已经进行了预处理)，所以现在的流程就是将NDC坐标系的梯度回传给世界坐标系。其次由于涉及到透视变换与可能存在的相机旋转，空间坐标系中任意一个方向的坐标变化($x$ 或 $y$ 或 $z$)都会引发NDC坐标系中$x$和$y$两个方向的变化，所以这里应该使用多元微分学中的**全导数公式**，即：
$$\begin{equation}
    \begin{aligned}
        \frac{\partial Loss}{\partial x_{3D}} &= \frac{\partial Loss}{\partial x_{ndc}} \cdot \frac{\partial x_{ndc}}{\partial x_{3D}} + \frac{\partial Loss}{\partial y_{ndc}} \cdot \frac{\partial y_{ndc}}{\partial x_{3D}} \\
        \frac{\partial Loss}{\partial y_{3D}} &= \frac{\partial Loss}{\partial x_{ndc}} \cdot \frac{\partial x_{ndc}}{\partial y_{3D}} + \frac{\partial Loss}{\partial y_{ndc}} \cdot \frac{\partial y_{ndc}}{\partial y_{3D}} \\
        \frac{\partial Loss}{\partial z_{3D}} &= \frac{\partial Loss}{\partial x_{ndc}} \cdot \frac{\partial x_{ndc}}{\partial z_{3D}} + \frac{\partial Loss}{\partial y_{ndc}} \cdot \frac{\partial y_{ndc}}{\partial z_{3D}} 
    \end{aligned}
\end{equation}
$$
* 现在需要求$\frac{\partial x_{ndc}}{\partial x_{3D}}$, $\frac{\partial y_{ndc}}{\partial x_{3D}}$, $\frac{\partial x_{ndc}}{\partial y_{3D}}$, $\frac{\partial y_{ndc}}{\partial y_{3D}}$, $\frac{\partial x_{ndc}}{\partial z_{3D}}$, $\frac{\partial y_{ndc}}{\partial z_{3D}}$
> MP变换：$$p_{hom} = P \cdot (x, y, z, 1)^T$$ $$p_x = P_{00}x + P_{01}y + P_{02}z + P_{03}$$  $$p_y = P_{10}x + P_{11}y + P_{12}z + P_{13}$$  $$p_w = P_{30}x + P_{31}y + P_{32}z + P_{33}$$  

$$\begin{aligned}
\frac{\partial x_{ndc}}{\partial x} &= \frac{\partial (\frac{p_x}{p_w})}{\partial x}
&= \frac{1}{p_w}\frac{\partial p_x}{\partial x} - \frac{p_x}{p_w^2}\frac{\partial p_w}{\partial x} 
&= \frac{P_{00}}{p_w} - \frac{p_x}{p_w^2} \cdot P_{30} \\
\frac{\partial y_{ndc}}{\partial x} &= \frac{\partial (\frac{p_y}{p_w})}{\partial x}
&= \frac{1}{p_w}\frac{\partial p_y}{\partial x} - \frac{p_y}{p_w^2}\frac{\partial p_w}{\partial x} 
&= \frac{P_{10}}{p_w} - \frac{p_y}{p_w^2} \cdot P_{30} \\
\frac{\partial x_{ndc}}{\partial y} &= \frac{\partial (\frac{p_x}{p_w})}{\partial y}
&= \frac{1}{p_w}\frac{\partial p_x}{\partial y} - \frac{p_x}{p_w^2}\frac{\partial p_w}{\partial y} 
&= \frac{P_{01}}{p_w} - \frac{p_x}{p_w^2} \cdot P_{31} \\
\frac{\partial y_{ndc}}{\partial y} &= \frac{\partial (\frac{p_y}{p_w})}{\partial y}
&= \frac{1}{p_w}\frac{\partial p_y}{\partial y} - \frac{p_y}{p_w^2}\frac{\partial p_w}{\partial y} 
&= \frac{P_{11}}{p_w} - \frac{p_y}{p_w^2} \cdot P_{31} \\
\frac{\partial x_{ndc}}{\partial z} &= \frac{\partial (\frac{p_x}{p_w})}{\partial z}
&= \frac{1}{p_w}\frac{\partial p_x}{\partial z} - \frac{p_x}{p_w^2}\frac{\partial p_w}{\partial z} 
&= \frac{P_{02}}{p_w} - \frac{p_x}{p_w^2} \cdot P_{32} \\
\frac{\partial y_{ndc}}{\partial z} &= \frac{\partial (\frac{p_y}{p_w})}{\partial z}
&= \frac{1}{p_w}\frac{\partial p_y}{\partial z} - \frac{p_y}{p_w^2}\frac{\partial p_w}{\partial z} 
&= \frac{P_{12}}{p_w} - \frac{p_y}{p_w^2} \cdot P_{32} \\
\end{aligned}$$
* 最后全部代入全导数公式即可：
$$\begin{equation}
    \begin{aligned}
        \frac{\partial Loss}{\partial \mu_{gaussian3d}}(\text{位置}) = 
        \begin{bmatrix}
            \left( \frac{P_{00}}{p_w} - P_{30}\frac{p_x}{p_w^2} \right) \cdot \frac{\partial Loss}{\partial x_{ndc}} + \left( \frac{P_{10}}{p_w} - P_{30}\frac{p_y}{p_w^2} \right) \cdot \frac{\partial Loss}{\partial y_{ndc}} \\
            \left( \frac{P_{01}}{p_w} - P_{31}\frac{p_x}{p_w^2} \right) \cdot \frac{\partial Loss}{\partial x_{ndc}} + \left( \frac{P_{11}}{p_w} - P_{31}\frac{p_y}{p_w^2} \right) \cdot \frac{\partial Loss}{\partial y_{ndc}} \\
            \left( \frac{P_{02}}{p_w} - P_{32}\frac{p_x}{p_w^2} \right) \cdot \frac{\partial Loss}{\partial x_{ndc}} + \left( \frac{P_{12}}{p_w} - P_{32}\frac{p_y}{p_w^2} \right) \cdot \frac{\partial Loss}{\partial y_{ndc}}
        \end{bmatrix}
    \end{aligned}
\end{equation}
$$

##### 由$\frac{\partial Loss}{\partial RGB_{gaussian2d}} $求$\frac{\partial Loss}{\partial \mu_{gaussian3d}}(\text{颜色})$
> 回顾球谐函数
> $$\begin{equation}
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
\end{equation}
> $$
* 由于这里3D均值通过影响归一化视角方向$\text{dir}$从而间接影响了RGB颜色，所以我们应该先利用**全导数法则**求$\frac{\partial Loss}{\partial \text{dir}}$:
    1. 首先明确归一化视角：
    $$\begin{equation}
        \begin{aligned}
            dir &= (dir_x, dir_y, dir_z) = \left( \frac{x'}{d}, \frac{y'}{d}, \frac{z'}{d} \right) \\
            &= \left( \frac{x'}{\sqrt{(x')^2 + (y')^2 + (z')^2}}, \frac{y'}{\sqrt{(x')^2 + (y')^2 + (z')^2}}, \frac{z'}{\sqrt{(x')^2 + (y')^2 + (z')^2}} \right) \\
            &=  \left(\frac{x - c_x}{\sqrt{(x - c_x)^2 + (y - c_y)^2 + (z - c_z)^2}}, \frac{y - c_y}{\sqrt{(x - c_x)^2 + (y - c_y)^2 + (z - c_z)^2}}, \frac{z - c_z}{\sqrt{(x - c_x)^2 + (y - c_y)^2 + (z - c_z)^2}} \right)
        \end{aligned}
    \end{equation}$$
    2. 分别展开求$\frac{\partial Loss}{\partial dir_x}$、$\frac{\partial Loss}{\partial dir_y}$、$\frac{\partial Loss}{\partial dir_z}$：
    $$
    \begin{equation}
        \begin{aligned}
            \frac{\partial Loss}{\partial x_{dir}} &= \frac{\partial Loss}{\partial Color_{RGB}} \cdot \frac{\partial Color_{RGB}}{\partial x_{dir}} \\
            &= \sum_{channel \in \{R,G,B\}} \frac{\partial Loss}{\partial Color_{channel}} \cdot \frac{\partial Color_{raw}}{\partial x} \\
            \frac{\partial Loss}{\partial y_{dir}} &= \frac{\partial Loss}{\partial Color_{RGB}} \cdot \frac{\partial Color_{RGB}}{\partial y_{dir}} \\
            &= \sum_{channel \in \{R,G,B\}} \frac{\partial Loss}{\partial Color_{channel}} \cdot \frac{\partial Color_{raw}}{\partial y} \\
            \frac{\partial Loss}{\partial z_{dir}} &= \frac{\partial Loss}{\partial Color_{RGB}} \cdot \frac{\partial Color_{RGB}}{\partial z_{dir}} \\
            &= \sum_{channel \in \{R,G,B\}} \frac{\partial Loss}{\partial Color_{channel}} \cdot \frac{\partial Color_{raw}}{\partial z}
        \end{aligned}
    \end{equation}
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
    $$\begin{equation}
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
    \end{equation}
    $$
* 最后利用矩阵的链式求导法则，得到:
  $$\begin{equation}
    \begin{aligned}
        \frac{\partial Loss}{\partial \mu_{gaussian3d}}(\text{颜色})
        &= (\frac{\partial dir}{\partial \mu_{gaussian3d}})^T \cdot \frac{\partial Loss}{\partial \text{dir}} \\
        &= \frac{1}{d} \left( I - dir \cdot dir^T \right) \cdot \frac{\partial Loss}{\partial \text{dir}} \\
        &= \frac{1}{d} \left( v_{grad} - dir \cdot (dir^T \cdot \frac{\partial Loss}{\partial \text{dir}}) \right)
    \end{aligned}
  \end{equation}
  $$

##### 由$\frac{\partial Loss}{\partial \Sigma^{-1}_{gaussian2d}}$求$\frac{\partial Loss}{\partial \mu_{gaussian3d}}(\text{形状})$
* 由于3D均值藏在 $J$ 矩阵中，所以需要根据 EWA Splatting 过程首先计算 $\frac{\partial Loss}{\partial J}$:
  $$\begin{equation}
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
  \end{equation}
  $$
> 这里用到了矩阵求导第一性原理：$$d(Loss) = \text{tr}\left( \left(\frac{\partial Loss}{\partial X}\right)^T dX \right)$$
* 从$\frac{\partial Loss}{\partial J}$ 回退到 $\frac{\partial Loss}{\partial (t_x, t_y, t_z)}$(相机坐标系下的3D均值)：
  $$\begin{equation}
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
  \end{equation}
  $$
* 由相机坐标系下的3D均值的梯度$\frac{\partial Loss}{\partial (t_x, t_y, t_z)}$回到世界坐标系下的均值的梯度 $\frac{\partial Loss}{\partial \mu_{gaussian3d}}(\text{形状})$
$$\begin{equation}
    \begin{aligned}
        &\because t_{cam} = (t_x, t_y, t_z) = W \cdot \mu_{gaussian3d} + t_{cam} \\
        &\therefore \frac{\partial Loss}{\partial \mu_{gaussian3d}}(\text{形状}) = W^T \cdot \frac{\partial Loss}{\partial (t_x, t_y, t_z)}
    \end{aligned}
  \end{equation}
  $$
> 经历了千辛万苦推到了这里，然而```CUDA```代码中并没有考虑这一部分，可能是为了节省算力考虑吧 $\dots$