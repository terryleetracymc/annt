package com.annt.trainning;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

import com.annt.network.RBMNetwork;

public class CDKBackPropagation implements Serializable {
	/**
	 * CD-k算法实现的反向传播训练 配合RBMNetwork进行训练
	 */
	private static final long serialVersionUID = 1195282112887327730L;

	public RBMNetwork rbm;
	public DoubleMatrix weight;
	public DoubleMatrix vbiass;
	public DoubleMatrix hbiass;
	// 权值更新，显层隐层偏置更新
	public DoubleMatrix wu;
	public DoubleMatrix vbu;
	public DoubleMatrix hbu;

	public int vn;
	public int hn;

	public int k = 1;

	public CDKBackPropagation(RBMNetwork n) {
		rbm = n;
		weight = rbm.weight;
		vbiass = rbm.vbiass;
		hbiass = rbm.hbiass;
		vn = rbm.vn;
		hn = rbm.hn;
	}

	public void setK(int kv) {
		k = kv;
	}

	// CD-k方法训练
	// 对单样本的更新
	public void updateMatrixAndBias(DoubleMatrix input) {
		// k迭代更新
		DoubleMatrix hi = null, vi = null;
		for (int i = 0; i < k; i++) {
			hi = getFeatureGivenSample(input);
			vi = getSampleGivenFeature(hi);
		}
		DoubleMatrix iOutput = rbm.getHOutput(input);
		DoubleMatrix sOutput = rbm.getHOutput(vi);
		// 可视层偏置向量更新
		vbu = input.sub(vi);
		// 隐藏层偏执向量更新
		hbu = iOutput.sub(sOutput);
		// 权值矩阵更新
		wu = input.mmul(iOutput.transpose()).sub(vi.mmul(sOutput.transpose()));
	}

	// 给定可视层输入采样隐藏层输出
	public DoubleMatrix getFeatureGivenSample(DoubleMatrix sample) {
		// 返回的隐藏层输出
		DoubleMatrix feature = new DoubleMatrix(hn);
		// 实际的RBM网络隐藏层输出
		DoubleMatrix aFeature = rbm.getHOutput(sample);
		// [0...1]正态分布随机数，用于采样分布
		DoubleMatrix deter = DoubleMatrix.rand(hn);
		// 采样获得样本
		for (int i = 0; i < hn; i++) {
			if (deter.get(i) < aFeature.get(i)) {
				feature.put(i, 1);
			} else {
				feature.put(i, 0);
			}
		}
		return feature;
	}

	// 给定隐藏层输入采样可视层输出
	public DoubleMatrix getSampleGivenFeature(DoubleMatrix feature) {
		// 返回的可视层输出
		DoubleMatrix sample = new DoubleMatrix(vn);
		// 实际的RBM网络可视层输出
		DoubleMatrix aSample = rbm.getVOutput(feature);
		// [0...1]正态分布随机数，用于采样分布
		DoubleMatrix deter = DoubleMatrix.rand(vn);
		// 采样获得样本
		for (int i = 0; i < vn; i++) {
			if (deter.get(i) < aSample.get(i)) {
				sample.put(i, 1);
			} else {
				sample.put(i, 0);
			}
		}
		return sample;
	}
}
