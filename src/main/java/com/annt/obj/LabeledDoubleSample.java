package com.annt.obj;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

public class LabeledDoubleSample implements Serializable {

	/**
	 * 有标记样本
	 */
	private static final long serialVersionUID = 3681097625558850399L;
	// 输入数据
	public DoubleMatrix input;
	// 理想输出
	public DoubleMatrix ideal;

	public LabeledDoubleSample(DoubleMatrix in, DoubleMatrix id) {
		input = in;
		ideal = id;
	}
}
