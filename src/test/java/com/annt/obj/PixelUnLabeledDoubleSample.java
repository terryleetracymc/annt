package com.annt.obj;

import org.jblas.DoubleMatrix;

public class PixelUnLabeledDoubleSample extends UnLabeledDoubleSample {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2311470367864287739L;

	int x;

	int y;

	public PixelUnLabeledDoubleSample(DoubleMatrix in, int px, int py) {
		super(in);
		x=px;
		y=py;
	}

}
