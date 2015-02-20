package com.annt.utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Scanner;

import org.jblas.DoubleMatrix;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;

public class CommonUtils {
	// 读json文本文件全文
	@SuppressWarnings("resource")
	public static String readJSONText(String path) throws FileNotFoundException {
		String content = new Scanner(new File(path)).useDelimiter("\\Z").next();
		return content;
	}

	// 读取数据集
	public static DoubleMatrix ReadDataset(String path)
			throws FileNotFoundException, IOException, ClassNotFoundException {
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(path));
		DoubleMatrix result = (DoubleMatrix) in.readObject();
		in.close();
		return result;
	}

	// 保存数据集
	public static void SaveDataset(String path, DoubleMatrix dataset)
			throws FileNotFoundException, IOException, ClassNotFoundException {
		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(
				path));
		out.writeObject(dataset);
		out.close();
	}

	// 存储图像到指定路径
	public static void SaveImgToPath(String path, double data[], int width,
			int height) {
		Mat img = new Mat(width, height, CvType.CV_64FC1);
		for (int m = 0; m < width; m++) {
			for (int n = 0; n < height; n++) {
				img.put(m, n, data[m + width * n]);
			}
		}
		Highgui.imwrite(path, img);
	}
}
