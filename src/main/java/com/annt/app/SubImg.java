package com.annt.app;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.opencv.core.Core;

import com.annt.utils.CommonUtils;

public class SubImg {
	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	public static void main(String[] args) {
		int idx = 0;
		for (idx = 0; idx < 23; idx++) {
			DoubleMatrix r1 = CommonUtils.ReadDataset("result/h27v06/r1mt/r"
					+ idx + ".mt");
			DoubleMatrix r2 = CommonUtils.ReadDataset("result/h27v06/o1mt/o"
					+ idx + ".mt");
			CommonUtils.SaveImgToPath("result/h27v06/sub" + idx + ".jpg",
					MatrixFunctions.abs(r1.sub(r2)).muli(255).toArray(), 4800,
					4800);
		}
	}

}
