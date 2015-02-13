package com.annt.evaluate;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class _1LEvaluate extends BasicEvaluate {

	/**
	 * 1L误差
	 */
	private static final long serialVersionUID = -3318750161713034998L;

	@Override
	public double getError(DoubleMatrix input, DoubleMatrix ideal) {
		DoubleMatrix result = MatrixFunctions.abs(input.sub(ideal));
		return result.mean();
	}

}
