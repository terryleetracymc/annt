package com.annt.debug;

import org.jblas.DoubleMatrix;

public class JBlasDebug {

	public static void main(String[] args) {
		DoubleMatrix m = DoubleMatrix.randn(5, 6);
		DoubleMatrix n = DoubleMatrix.randn(6, 5);
		System.out.println(m.mmul(n));
	}

}
