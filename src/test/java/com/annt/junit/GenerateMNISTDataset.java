package com.annt.junit;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

import org.jblas.DoubleMatrix;

import com.annt.utils.CommonUtils;

public class GenerateMNISTDataset {
	public static void main(String[] args) throws IOException,
			ClassNotFoundException {
		DataInputStream images = new DataInputStream(new FileInputStream(
				"/Users/terry/Desktop/train-images-idx3-ubyte"));
		int magicNumber = images.readInt();
		if (magicNumber != 2051) {
			System.err.println("Image file has wrong magic number: "
					+ magicNumber + " (should be 2051)");
			System.exit(0);
		}
		int numImages = images.readInt();
		int numRows = images.readInt();
		int numCols = images.readInt();

		long start = System.currentTimeMillis();
		int numImagesRead = 0;
		DoubleMatrix dataset = new DoubleMatrix(784, 60000);
		while (images.available() > 0 && numImagesRead < numImages) {
			System.out.println("读取第" + (numImagesRead + 1) + "张图像");
			double[][] image = new double[numCols][numRows];
			for (int colIdx = 0; colIdx < numCols; colIdx++) {
				for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
					image[colIdx][rowIdx] = images.readUnsignedByte();
				}
			}
			DoubleMatrix sample = new DoubleMatrix(image);
			dataset.putColumn(numImagesRead, sample);
			numImagesRead++;
		}
		System.out.println();
		CommonUtils.SaveDataset("/Users/terry/Desktop/mnist.dat", dataset);
		long end = System.currentTimeMillis();
		long elapsed = end - start;
		long minutes = elapsed / (1000 * 60);
		long seconds = (elapsed / 1000) - (minutes * 60);
		System.out.println("Read " + numImagesRead + " samples in " + minutes
				+ " m " + seconds + " s ");
		images.close();
	}

}