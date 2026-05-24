/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
// 计算高斯椭球颜色
// Forward method for converting the input spherical harmonics 
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// campos 即 camera position 是相机位置：用于后期计算光线方向
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	// 观察方向
	glm::vec3 pos = means[idx];
	glm::vec3 dir =   - campos;
	dir = dir / glm::length(dir);
	// 找到 thread 对应的sh系数(可以看到一组系数是由三个系数构成的，分别是RGB三个颜色的系数)
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	// 0阶 球谐函数 (基础颜色 / 漫反射)
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		// 三个方向 (基础的光照变化)
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		// 1阶 球谐函数 (更复杂的环境光反射)
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			// 2阶 球谐函数
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				// 3阶 球谐函数 (极高频的高光细节)
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;
	// 因为球谐函数算出来的结果是零均值的（有正有负）
	// 但在计算机图形学中，RGB 颜色是从 0 到 1 的
	// 所以作者在训练时规定了一个 +0.5 的偏移量，将其映射到正常的正数颜色区间。
	
	// clamp操作类似神经网络中的 ReLU 激活函数
	// 将颜色小于 0 的 通道(RGB channels) 强制变成了 0
	// 反向传播计算梯度时，这个位置的梯度也应该被切断（不让它继续更新）
	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// 计算二维协方差矩阵
// !! 这里是最核心的splat的过程 !!
// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	// 获得 3d gaussian 椭球球心 在相机坐标系下的坐标
	float3 t = transformPoint4x3(mean, viewmatrix);
	// 限制相机的水平和垂直方向
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	// 高斯椭球在相机坐标系下的视角要小于1.3倍视角 
	// 这里的理解就是不要高斯椭球不能太偏
	// 强行把那些过于偏离视线中心的高斯体给“拉”回到安全的数学边界内，这被称为 Ray Clamping 技巧
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	// 定义投影变换的jacobi矩阵
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);
	// 定义 V 矩阵
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);
	// 完成 MVJ 的过程
	glm::mat3 T = W * J;
	// 获取3d gaussian 椭球的协方差矩阵 (用已有的6个参数进行还原)
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);
	// 对协方差矩阵进行Splat(MVJ)操作
	// 仍然因为GLM库在底层内存是列主序，而C++数组是行主序
	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	// GLM (OpenGL Mathematics)是一个在计算机图形学中极其著名的 C++ 数学库。
	// 在 3D 渲染中，我们需要极其频繁地处理向量、矩阵乘法、四元数等操作，
	// 而原生的 C++ 或 CUDA 并没有自带这些好用的数学结构
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;
	// mod = scale_modifier。
	// 在 3DGS 的训练早期，为了防止高斯球长得太大互相遮挡，通常会传入一个小于 1 的 mod 强行把所有高斯球缩小一点。

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);
	// \Sigma = R * S * S_T * R_T 
	//        = R_T * S_T * S * R (GPU列主序)
	// 		  = M_T * M (M = S * R)
	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}
// preprocessCUDA的kernel函数 就是一个 splat + 筛选 的过程:
// 将场景中一个个 3D gaussian 球 过滤 - splat - 过滤

// 最终得到的是一个个 2D gaussian 椭圆(分布) 的种种性质
// 被 splat 到屏幕上的哪个 tile 上
// splat后的2d gaussian 分布如何
// 每一个2d 椭圆对应在tile上的深度如何

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	// 输出 output
	int* radii, // 2D 最大包围半径
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity, // 2D 协方差的逆矩阵参数和最终的不透明度，是像素渲染时计算高斯衰减的灵魂数据
	const dim3 grid,
	uint32_t* tiles_touched, //高斯体到底覆盖了几个 $16 \times 16$ 的屏幕图块
	bool prefiltered,
	bool antialiasing)
{
	auto idx = cg::this_grid().thread_rank();
	// 常用操作!!!!!
	// 这里利用了cooperative_groups(协同组，通常在代码中简写为 cg)。
	// 对应了老代码的:int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// 在这里，算出来的 idx 就是当前这个线程负责处理的那个三维高斯体在数组中的索引位置。
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view; // 相机坐标系
	// float3 是一个内置的三维向量数据类型，主要用于表示和处理三维空间中的坐标、向量、颜色等需要三个浮点数分量的数据。
	// 这里判断点云在不在视锥范围以内 + “世界坐标系 -> 相机坐标系”
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	// 做投影操作
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] }; //世界坐标系
	float4 p_hom = transformPoint4x4(p_orig, projmatrix); // 进行 V * P变换 -> 在齐次剪切空间中的坐标
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
		// 一个 $3 \times 3$ 的协方差矩阵是对称的，所以只需要存 6 个独立的浮点数。
	}
	else
	{
		// 通过 旋转因子(3) 和 缩放因子(4) 重构 3D 高斯分布的协方差矩阵
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
		// cov3Ds + idx * 6 :在全局数组中，精准定位到当前线程专属的那 6 个格子的内存起始地址。
	}

	// Compute 2D screen-space covariance matrix
	// 使用EWA Splatting的方法计算协方差的 2D 投影
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	constexpr float h_var = 0.3f;
	// 引入 h_var 当作低通滤波器,避免 2d gaussian方差过小产生 aliasing(走样)
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	// 未进行低通滤波的行列式 (可以理解为2d gaussian椭圆的面积)
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	// 进行低通滤波后的行列式 (可以理解为被撑胖后的2d gaussian椭圆的面积)
	float h_convolution_scaling = 1.0f;

	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability
		// 这里作者说是为了数值稳定性
		// 这里如果开启了 antialiasing（反走样），
		// 这个 h_convolution_scaling 会在稍后乘以该高斯体的不透明度 (opacity)
		// 效果 : 
		// 远处的点虽然被强行放大了，但它们会变得更加透明，整体的光照能量保持不变，画面极为平滑。
		// !!!! Mip-Splatting !!!!
	// Invert covariance (EWA algorithm)
	const float det = det_cov_plus_h_cov;

	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
	// 二维 gaussian 分布 : $f(x) = \exp(-\frac{1}{2} X^T \Sigma^{-1} X)$
	// 这里为了算 -\frac{1}{2} X^T \Sigma^{-1} X 

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	// 首先计算得到 \sigma
	// 而后利用 3/sigma 原则 确定保留的边界
	// 由于屏幕的像素坐标系是整数, 所以利用ceil向上取整获得
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	// ndc2Pix 将归一化设备坐标系映射到真实的屏幕像素坐标系
	// point_image 为 2d 高斯椭圆中心点在像素坐标系上的真是位置
	uint2 rect_min, rect_max;
	// 通过Bounding Box获得高斯椭圆的包围盒
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	// !!! 需要看一下 getRect 这个函数 !!!
	// getRect的作用就是为了算这个椭圆包围盒的 左上 和 右下
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;
	// 这个高斯球虽然在相机的正前方（没有被前面的视锥剔除），
	// 但它实在偏得太离谱了，根本没出现在屏幕画面里！

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}
	// 打包入库以备后续使用
	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	// 存radii的作用
	// 在后续 renderCUDA 中，对于一个Tile内的像素, 用这个 radii 结合高斯中心坐标画一个包围盒，
	// 如果像素在这个 radii 之外，就直接跳过不作计算。
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	float opacity = opacities[idx];


	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };


	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// 最终渲染的函数
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth)
// __launch_bounds__(BLOCK_X * BLOCK_Y) :
// 向 nvcc 编译器下达每个 block 中 thread 数不会超过BLOCK_X(16) * BLOCK_Y(16)的命令
// 作用：精准地算出在 256 个 threads 的限制下，每个线程最多可以使用多少个寄存器，优化代码速度
// __restrict__ 告诉 nvcc 编译器 指针指向的内存区域绝对不会互相重叠，许 GPU 开启只读缓存加速 (Read-Only Cache)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	// 通过向上取整的方式求共有多少blocks
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y }; // 当前 Tile 左上角的像素坐标
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) }; // 当前 Tile 右下角的像素坐标
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };  // thread 像素坐标
	uint32_t pix_id = W * pix.y + pix.x; // 像素的索引
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	// 这里需要明确 uint2 是 一个 CUDA 内置容器
	// 本质上就是一个包含了两个无符号整数的 C++ 结构体
	// 它在底层默认把这两个数字命名为 x 和 y
	// 所以就能够弄清楚 ranges 里面存了当前 grid 下所有 block 所对应的 range 
	// 这每一组range = (start, end) 其实是负责查找 point_list 中对应该tile的 2d gaussian椭球的
	// 现在这一步本质上就是查找当前 block(tile) 对应的那个 point_list区间
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	// 为共享内存做准备
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	// GPU中的共享内存机制
	// __shared__ 关键字
	// 只要变量前面加了它，这个变量就会被分配在共享内存中。
	// 同一个 Block（线程块）内的所有线程（即我们这个 Tile 里的 256 个像素线程）
	// 可以共同读写这块内存，并且互相可见。

	// 引入共享内存是为了解决 内存读取拥堵 问题 详细解释看笔记
	// Initialize helper variables
	float T = 1.0f; // 透射率
	uint32_t contributor = 0; // 多少个高斯椭球参与贡献
	uint32_t last_contributor = 0; // 从前往后最后一个参与贡献的高斯椭球是谁
	float C[CHANNELS] = { 0 }; // color

	float expected_invdepth = 0.0f; // 期望逆深度， 用于渲染出场景深度图

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		// __syncthreads_count(predicate)
		// 发起一次“全员举手有关 predict 的投票”
		// 它会让当前 Block 里的所有线程停下来，统计有多少个线程的 done 变量是 true
		// 返回总数
		if (num_done == BLOCK_SIZE)
			break;
			// prune 操作 减小渲染压力
		
		// 这一部分还是每一个 thread 在操作
		// 每一个thread 从 point_list 中取出一个 gaussian 参数
		// 然后存放到共享内存中，准备后续使用
		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			// block.thread_rank() : 
			// 获取当前线程在这个 Block 内部的一维线性编号 (0 ~ 255)
			
		}
		block.sync();
		// 同步屏障
		// == __syncthreads()
		// 竞态条件

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;
				// 异常检查 power 必须为负

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power)); // 取0，99作为最大值是为了后续的梯度下降
			if (alpha < 1.0f / 255.0f)
				continue;
				// 显示器为 8-bit (RGB * 0-255) 
				// 如果当前这个高斯体对这个像素的贡献度 alpha 连 1/255 都不到
				// 意味着它对最终颜色的改变根本无法被肉眼看见
			// 当前的能看到此 gaussian 的透射率 test_T
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
				// feature[collected_id[j] * CHANNELS + ch]
				// 表示第 j 个高斯球通过SH算出来的在接触点的第ch个颜色通道

			if(invdepth)
			expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
			// 在普通的图像处理(OpenCV)中，像素颜色通常是交织排列的：RGB RGB RGB...（这叫 HWC 格式）。
			// 在深度学习（PyTorch）中，张量（Tensor）的标准排布格式是：先把所有的 R 通道写完，再写所有的 G，最后写所有的 B（这叫 CHW 格式）
			
			if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* depths,
	float* depth)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths, 
		depth);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing
		);
}
