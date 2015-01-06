package com.annt.obj;

import org.jblas.DoubleMatrix;

public class DoubleLabelSample extends LabelSample {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8310373362253244102L;
	public DoubleMatrix data;
	public int label;

	public DoubleLabelSample(DoubleMatrix d, int l) {
		data = d;
		label = l;
	}
}
