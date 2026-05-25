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
        \frac{\partial Loss}{\partial \mu_{gaussian3d}} = 
        \begin{bmatrix}
            \left( \frac{P_{00}}{p_w} - P_{30}\frac{p_x}{p_w^2} \right) \cdot \frac{\partial Loss}{\partial x_{ndc}} + \left( \frac{P_{10}}{p_w} - P_{30}\frac{p_y}{p_w^2} \right) \cdot \frac{\partial Loss}{\partial y_{ndc}} \\
            \left( \frac{P_{01}}{p_w} - P_{31}\frac{p_x}{p_w^2} \right) \cdot \frac{\partial Loss}{\partial x_{ndc}} + \left( \frac{P_{11}}{p_w} - P_{31}\frac{p_y}{p_w^2} \right) \cdot \frac{\partial Loss}{\partial y_{ndc}} \\
            \left( \frac{P_{02}}{p_w} - P_{32}\frac{p_x}{p_w^2} \right) \cdot \frac{\partial Loss}{\partial x_{ndc}} + \left( \frac{P_{12}}{p_w} - P_{32}\frac{p_y}{p_w^2} \right) \cdot \frac{\partial Loss}{\partial y_{ndc}}
        \end{bmatrix}
    \end{aligned}
\end{equation}
$$

##### 由$\frac{\partial Loss}{\partial RGB_{gaussian2d}} $求$\frac{\partial Loss}{\partial \mu_{gaussian3d}}(\text{位置})$
* 由于这里3D均值通过影响视角方向从而间接影响了RGB颜色，所以我们应该先求$\frac{\partial Loss}{\partial \text{dir}}$
> 回顾球谐函数