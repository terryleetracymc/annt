package com.annt.obj;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

public class DoubleSample implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3681097625558850399L;
	// 输入数据
	public DoubleMatrix input;
	// 理想输出
	public DoubleMatrix ideal;

	public DoubleSample(DoubleMatrix in, DoubleMatrix id) {
		input = in;
		ideal = id;
	}
}
