package com.annt.junit;

import java.io.FileNotFoundException;
import java.io.IOException;

import org.jblas.DoubleMatrix;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;

import com.annt.utils.CommonUtils;

public class OpenCVTest {
	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	public static void main(String args[]) throws FileNotFoundException,
			ClassNotFoundException, IOException {
		DoubleMatrix dataset = CommonUtils
				.ReadDataset("/Users/terry/Desktop/MNIST.dat");
		for (int i = 0; i < 1000; i++) {
			System.out.println("处理" + i + "图像");
			DoubleMatrix sample = dataset.getColumn(i);
			Mat img = new Mat(28, 28, CvType.CV_64FC1);
			for (int m = 0; m < 28; m++) {
				for (int n = 0; n < 28; n++) {
					double pixel = sample.get(m + 28 * n);
					img.put(m, n, pixel);
				}
			}
			Highgui.imwrite("images/origin/" + i + ".jpg", img);
			img.empty();
		}
		
		System.out.println("Done");
	}

}