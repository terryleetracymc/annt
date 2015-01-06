package com.annt.debug;

import org.jblas.DoubleMatrix;


public class JBlasDebug {

	public static void main(String[] args) {
		DoubleMatrix a=DoubleMatrix.randn(5000, 5000);
		DoubleMatrix b=DoubleMatrix.randn(5000, 5000);
		System.out.println("start");
		a.mmul(b);
		System.out.println("success");
	}
	
}
