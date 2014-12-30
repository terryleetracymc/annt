package com.annt.interf;

import java.util.LinkedList;

import org.jblas.DoubleMatrix;

public interface NetOperation {

	// 获得激活值
	LinkedList<DoubleMatrix> getActives(DoubleMatrix input);

	// 获得输出值
	LinkedList<DoubleMatrix> getOutputs(DoubleMatrix input);

	//
	DoubleMatrix getOutput(DoubleMatrix input);

	// 权值更新函数
	void updateNet(LinkedList<DoubleMatrix> w, LinkedList<DoubleMatrix> b,
			double learning_rate);
}
