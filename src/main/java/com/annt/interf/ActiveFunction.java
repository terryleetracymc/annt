package com.annt.interf;

import org.jblas.DoubleMatrix;

/**
 * 
 * @author terry 激活函数接口
 */
public interface ActiveFunction {

	// 输出值
	DoubleMatrix output(DoubleMatrix w, DoubleMatrix x, DoubleMatrix b);

	// 输出值，为减少一些激活函数的计算量特别设置的接口
	DoubleMatrix output(DoubleMatrix x);

	// 激活数值
	DoubleMatrix active(DoubleMatrix w, DoubleMatrix x, DoubleMatrix b);

	// 激活函数导数
	DoubleMatrix derivative(DoubleMatrix w, DoubleMatrix x, DoubleMatrix b);

	// 激活函数导数，为减少一些激活函数的求导计算量特别设置的接口
	DoubleMatrix derivative(DoubleMatrix x);

}
