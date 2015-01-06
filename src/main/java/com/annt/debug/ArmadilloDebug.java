package com.annt.debug;

import org.armadillojava.Mat;

public class ArmadilloDebug {

	public static void main(String[] args) {
		Mat a = new Mat(5000, 5000);
		a.randn();
		Mat b = new Mat(5000, 5000);
		b.randn();
		System.out.println("start");
		a.times(b);
		System.out.println("success");
	}

}
