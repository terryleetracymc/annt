package com.annt.obj;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

public class LabeledDoubleSample implements Serializable {

	/**
	 * 有标记样本
	 */
	private static final long serialVersionUID = 3681097625558850399L;
	// 输入数据
	public DoubleMatrix data;
	// 输出样本开始位置
	public int oidx;

	public LabeledDoubleSample(DoubleMatrix in, int i) {
		data = in;
		oidx = i;
	}
}
