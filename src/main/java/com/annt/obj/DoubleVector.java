package com.annt.obj;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

public class DoubleVector implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4732939110746197079L;

	DoubleMatrix data;

	public DoubleVector(DoubleMatrix d) {
		data = d;
	}
}
