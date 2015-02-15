package com.annt.junit;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import com.annt.app.BaseApp;
import com.annt.evaluate._2LEvaluate;
import com.annt.network.RBMNetwork;
import com.annt.network.SimpleNetwork;
import com.annt.trainning.CDKBackPropagation;

public class StandaloneApp extends BaseApp {

	/**
	 * 单机模式
	 */
	private static final long serialVersionUID = -4140616509724444832L;

	// 读取数据集
	public static DoubleMatrix readDataset(String path)
			throws FileNotFoundException, IOException, ClassNotFoundException {
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(path));
		DoubleMatrix result = (DoubleMatrix) in.readObject();
		in.close();
		return result;
	}

	public void getTargetRBM(DoubleMatrix dataset, RBMNetwork rbm,
			double max_error, int time, double learning_rate, int cd_k) {
		DoubleMatrix sample = null;
		CDKBackPropagation cdkBP = new CDKBackPropagation(rbm);
		cdkBP.setK(cd_k);
		for (int m = 0; m < time; m++) {
			DoubleMatrix wu = DoubleMatrix.zeros(rbm.vn, rbm.hn);
			DoubleMatrix vu = DoubleMatrix.zeros(rbm.vn);
			DoubleMatrix hu = DoubleMatrix.zeros(rbm.hn);
			// 获得权值更新
			for (int i = 0; i < dataset.columns; i++) {
				sample = dataset.getColumn(i);
				cdkBP.updateMatrixAndBias(sample);
				wu.addi(cdkBP.wu);
				vu.addi(cdkBP.vbu);
				hu.addi(cdkBP.hbu);
			}
			// 平均权值更新
			wu.divi(dataset.columns);
			vu.divi(dataset.columns);
			hu.divi(dataset.columns);
			rbm.updateRBM(wu, vu, hu, learning_rate);
			// 计算误差
			double error = 0.0;
			_2LEvaluate evaluate = new _2LEvaluate();
			for (int i = 0; i < dataset.columns; i++) {
				sample = dataset.getColumn(i);
				DoubleMatrix restoreSign = rbm.getVOutput(rbm
						.getHOutput(sample));
				error += evaluate.getError(sample, restoreSign);
			}
			// System.out.println(error);
			if (error / dataset.columns < max_error) {
				break;
			}
		}
	}

	@Test
	public void testRBM() throws FileNotFoundException, ClassNotFoundException,
			IOException {
		DoubleMatrix dataset = readDataset("/Users/terry/Desktop/dts.dat");
		RBMNetwork rbm = new RBMNetwork(25, 20, 100);
		getTargetRBM(dataset, rbm, 0.2, 5, 0.5, 5);
		SimpleNetwork firstNetwork = rbm.getNetwork();
		SimpleNetwork secondNetwork = rbm.getRNetwork();
		firstNetwork.addUpperNetwork(secondNetwork);
		int idx = 15;
		System.out.println(dataset.getColumn(idx));
		System.out.println(firstNetwork.getOutput(dataset.getColumn(idx)));
	}
}
