package com.annt.junit;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import com.annt.app.BaseApp;
import com.annt.evaluate._2LEvaluate;
import com.annt.network.RBMNetwork;
import com.annt.obj.PixelUnLabeledDoubleSample;
import com.annt.trainning.CDKBackPropagation;

public class StandaloneApp extends BaseApp {

	/**
	 * 单机模式
	 */
	private static final long serialVersionUID = -4140616509724444832L;

	// RBM训练
	public static RBMNetwork RBMTrainning(
			ArrayList<PixelUnLabeledDoubleSample> dataset, int vn, int hn,
			double learning_rate) {
		RBMNetwork rbm = new RBMNetwork(vn, hn, 100);
		CDKBackPropagation cdkbp = new CDKBackPropagation(rbm);
		for (int m = 0; m < 10; m++) {
			//
			DoubleMatrix updateMatrix = DoubleMatrix.zeros(vn, hn);
			DoubleMatrix updateVbiass = DoubleMatrix.zeros(vn);
			DoubleMatrix updateHbiass = DoubleMatrix.zeros(hn);
			// 获得更新权值
			for (int i = 0; i < dataset.size(); i++) {
				cdkbp.updateMatrixAndBias(dataset.get(i).input);
				updateMatrix.addi(cdkbp.wu);
				updateVbiass.addi(cdkbp.vbu);
				updateHbiass.addi(cdkbp.hbu);
			}
			// 平均化权值
			updateMatrix.divi(dataset.size());
			updateVbiass.divi(dataset.size());
			updateHbiass.divi(dataset.size());
			// 使用权值更新RBM网络权值结构
			rbm.updateRBM(updateMatrix, updateVbiass, updateHbiass,
					learning_rate);
			// 看信号的还原值
			double error = 0.0;
			_2LEvaluate evalute = new _2LEvaluate();
			for (int i = 0; i < dataset.size(); i++) {
				DoubleMatrix restoreSign = rbm.getVOutput(rbm
						.getHOutput(dataset.get(i).input));
				error = error
						+ evalute.getError(dataset.get(i).input, restoreSign);
			}
			// System.out.println(error / dataset.size());
		}
		return rbm;
	}

	@SuppressWarnings("unchecked")
	@Test
	public void testRBM() throws FileNotFoundException, IOException,
			ClassNotFoundException {
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				"/Users/terry/Desktop/dts.dat"));
		ArrayList<PixelUnLabeledDoubleSample> dataset = (ArrayList<PixelUnLabeledDoubleSample>) in
				.readObject();
		RBMNetwork target = RBMTrainning(dataset, 25, 20, 2);
		int idx = 255;
		System.out.println(dataset.get(idx).input);
		System.out
				.println(target.getVOutput(target.getHOutput(dataset.get(idx).input)));
		in.close();
	}
}
