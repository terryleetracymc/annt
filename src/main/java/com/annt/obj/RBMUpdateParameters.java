package com.annt.obj;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

import com.annt.network.RBMNetwork;

public class RBMUpdateParameters implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3944202517703158149L;

	// 权值矩阵更新
	public DoubleMatrix wu;

	// 隐藏层偏置更新
	public DoubleMatrix hu;

	// 可视层偏置更新
	public DoubleMatrix vu;

	// rbm网络
	public RBMNetwork rbm;

	// 构造函数
	public RBMUpdateParameters(RBMNetwork rbmNetwork) {
		rbm = rbmNetwork;
		hu = DoubleMatrix.zeros(rbm.hn);
		vu = DoubleMatrix.zeros(rbm.vn);
		wu = DoubleMatrix.zeros(rbm.vn, rbm.hn);
	}

	// 清零
	public void zeroAll() {
		hu.subi(hu);
		vu.subi(vu);
		wu.subi(wu);
	}

	// 添加权值衰减参数
	public void addLamdaWeight(double lamda, DoubleMatrix w) {
		wu.addi(w.mul(lamda));
	}

	// 添加更新参数
	public void addAll(DoubleMatrix ov, DoubleMatrix oh, DoubleMatrix ow) {
		vu.addi(ov);
		hu.addi(oh);
		wu.addi(ow);
	}

	// 平均更新参数
	public void div(long datasetSize) {
		vu.divi(datasetSize);
		hu.divi(datasetSize);
		wu.divi(datasetSize);
	}
}
