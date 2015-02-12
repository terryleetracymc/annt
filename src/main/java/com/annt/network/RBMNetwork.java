package com.annt.network;

import org.jblas.DoubleMatrix;

import com.annt.function.SigmoidFunction;

public class RBMNetwork extends BasicNetwork {

	/**
	 * 基于受限玻尔兹曼机的网络结构实现
	 */
	private static final long serialVersionUID = 2904054327850971298L;

	// 显层和隐层神经元个数
	public int vn, hn;
	// 连接显层和隐层的权值矩阵
	public DoubleMatrix weight;
	// 显层偏置
	public DoubleMatrix vbiass;
	// 隐层偏置
	public DoubleMatrix hbiass;
	// sigmoid函数
	public SigmoidFunction sigmoid = new SigmoidFunction();

	// 更新RBM网络参数
	public void updateRBM(DoubleMatrix wu, DoubleMatrix vbu, DoubleMatrix hbu,
			double learning_rate) {
		weight.addi(wu.mul(learning_rate));
		vbiass.addi(vbu.mul(learning_rate));
		hbiass.addi(hbu.mul(learning_rate));
	}

	// 初始化RBM网络结构
	public RBMNetwork(int v, int h, int divRatio) {
		vn = v;
		hn = h;
		weight = DoubleMatrix.rand(vn, hn).div(divRatio);
		// 根据给定的样本集确定可视层的偏置数值
		// 待实现
		vbiass = DoubleMatrix.rand(vn).div(divRatio);
		hbiass = DoubleMatrix.zeros(hn);
	}

	//
	public RBMNetwork(int v, int h, int divRatio, DoubleMatrix datasets) {
		vn = v;
		hn = h;
		weight = DoubleMatrix.rand(vn, hn).div(divRatio);
		// ******************
		vbiass = DoubleMatrix.rand(vn).div(divRatio);
		hbiass = DoubleMatrix.zeros(hn);
	}

	// 得到隐藏层输出
	public DoubleMatrix getHOutput(DoubleMatrix input) {
		return sigmoid.output(weight, input, hbiass);
	}

	// 得到可视层输出
	public DoubleMatrix getVOutput(DoubleMatrix feature) {
		return sigmoid.output(weight.transpose(), feature, vbiass);
	}
}
