package com.annt.app;

import org.jblas.DoubleMatrix;
import org.opencv.core.Core;

import com.annt.utils.CommonUtils;

public class Test {
	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	public static void main(String[] args) {
		int idx = 1;
		for (idx = 0; idx < 23; idx++) {
			DoubleMatrix result = CommonUtils.ReadDataset("result/h27v06/"
					+ idx + ".mt");
			result.muli(255);
			CommonUtils.SaveImgToPath("result/h27v06/" + idx + ".jpg",
					result.toArray(), 4800, 4800);
		}
	}
}
