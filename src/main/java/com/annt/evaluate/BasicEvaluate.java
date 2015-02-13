package com.annt.evaluate;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

abstract public class BasicEvaluate implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6130845101435788516L;

	abstract public double getError(DoubleMatrix input, DoubleMatrix ideal);
}
