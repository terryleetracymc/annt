package com.annt.network;

import org.jblas.DoubleMatrix;

import com.annt.function.SigmoidFunction;

public class RBMNetwork extends BasicNetwork {

	/**
	 * 基于受限玻尔兹曼机的网络结构实现
	 */
	private static final long serialVersionUID = 2904054327850971298L;

	// 显层和隐层神经元个数
	int vn, hn;
	// 连接显层和隐层的权值矩阵
	DoubleMatrix weight;
	// 显层偏置
	DoubleMatrix vbiass;
	// 隐层偏置
	DoubleMatrix hbiass;
	// sigmoid函数
	SigmoidFunction sigmoid = new SigmoidFunction();

	// 初始化RBM网络结构
	public RBMNetwork(int v, int h) {
		vn = v;
		hn = h;
		weight = DoubleMatrix.zeros(vn, hn);
		vbiass = DoubleMatrix.zeros(vn);
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
