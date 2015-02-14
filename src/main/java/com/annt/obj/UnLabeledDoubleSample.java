package com.annt.obj;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

public class UnLabeledDoubleSample implements Serializable {

	/**
	 * 无标记样本
	 */
	private static final long serialVersionUID = -1074832445779047544L;
	// 输入数据
	public DoubleMatrix input;

	public UnLabeledDoubleSample(DoubleMatrix in) {
		input = in;
	}
}
