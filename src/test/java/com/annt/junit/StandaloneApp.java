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
import com.annt.obj.UpdateParameters;
import com.annt.trainning.CDKBackPropagation;
import com.annt.trainning.SimpleBackPropagation;

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

	// 根据样本使用BP算法训练自编码器神经网络
	public void getTargetNetwork(DoubleMatrix dataset, SimpleNetwork network,
			double max_error, int time, double learning_rate, double lamda) {
		DoubleMatrix sample = null;
		SimpleBackPropagation sbp = new SimpleBackPropagation(network);
		UpdateParameters ups = new UpdateParameters(network);
		ups.zeroAll();
		for (int m = 0; m < time; m++) {
			for (int i = 0; i < dataset.columns; i++) {
				sample = dataset.getColumn(i);
				sbp.updateMatrixAndBias(sample, sample);
				ups.addAll(sbp.weights_updates, sbp.biass_updates);
			}
			ups.div(dataset.columns);
			ups.addLamdaWeights(lamda);
			network.updateNet(ups.wus, ups.bus, learning_rate);
			// 计算误差
			double error = 0.0;
			_2LEvaluate evaluate = new _2LEvaluate();
			for (int i = 0; i < dataset.columns; i++) {
				sample = dataset.getColumn(i);
				error += evaluate.getError(sample, network.getOutput(sample));
			}
			System.out.println(error / dataset.columns);
			if (error / dataset.columns < max_error) {
				break;
			}
		}
	}

	// 根据样本使用CD-K算法初始化RBM
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
			System.out.println(error / dataset.columns);
			if (error / dataset.columns < max_error) {
				break;
			}
		}
	}

	// 使用RBM生成第一层网络
	// @Test
	public void RBMGenerateL1() throws FileNotFoundException,
			ClassNotFoundException, IOException {
		int idx = 100;
		DoubleMatrix dataset = readDataset("/Users/terry/Desktop/dts.dat");
		RBMNetwork rbm = new RBMNetwork(25, 15, 100);
		getTargetRBM(dataset, rbm, 0.3, 30, 0.5, 2);
		SimpleNetwork firstNetwork = rbm.getNetwork();
		// 存储第一层特征提取层网络
		SimpleNetwork.saveNetwork("rbm/ts_l1", firstNetwork);
		SimpleNetwork secondNetwork = rbm.getRNetwork();
		firstNetwork.addUpperNetwork(secondNetwork);
		System.out.println(dataset.getColumn(idx));
		System.out.println(firstNetwork.getOutput(dataset.getColumn(idx)));
		// 存储特征提取层网络以及信号恢复层网络
		SimpleNetwork.saveNetwork("rbm/ts_l1r.nt", firstNetwork);
	}

	// 载入RBM第一层模型使用BP
	// @Test
	public void L1Training() throws FileNotFoundException,
			ClassNotFoundException, IOException {
		DoubleMatrix dataset = readDataset("/Users/terry/Desktop/dts.dat");
		SimpleNetwork ts_l1r = SimpleNetwork
				.loadNetwork("network/ts_l1r_g_2.nt");
		getTargetNetwork(dataset, ts_l1r, 0.1, 10000, 0.8, 0.5);
		SimpleNetwork.saveNetwork("network/ts_l1r_g_1.nt", ts_l1r);
	}

	// 查看信号恢复情况
	@Test
	public void SeeRestoreSign() throws FileNotFoundException,
			ClassNotFoundException, IOException {
		DoubleMatrix dataset = readDataset("/Users/terry/Desktop/dts_all.dat");
		SimpleNetwork ts_l1r = SimpleNetwork
				.loadNetwork("network/ts_l1r_g_1.nt");
		int idx = 8000;
		System.out.println(dataset.getColumn(idx));
		System.out.println(ts_l1r.getOutput(dataset.getColumn(idx)));
	}
}
