package com.annt.evaluate;

import org.jblas.DoubleMatrix;

public class _2LEvaluate extends BasicEvaluate {

	/**
	 * 2L误差
	 */
	private static final long serialVersionUID = 3549104291806233916L;

	@Override
	public double getError(DoubleMatrix input, DoubleMatrix ideal) {
		DoubleMatrix result = input.sub(ideal);
		return result.norm2();
	}

}
