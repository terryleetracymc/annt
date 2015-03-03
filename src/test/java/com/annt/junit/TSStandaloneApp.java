package com.annt.junit;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.LinkedList;
import java.util.Random;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import com.annt.app.BaseApp;
import com.annt.evaluate._2LEvaluate;
import com.annt.network.RBMNetwork;
import com.annt.network.SimpleNetwork;
import com.annt.obj.NetworkUpdateParameters;
import com.annt.trainning.CDKBackPropagation;
import com.annt.trainning.SimpleBackPropagation;

public class TSStandaloneApp extends BaseApp {

	/**
	 * 单机模式
	 */
	private static final long serialVersionUID = -4140616509724444832L;

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

	// 根据样本使用BP算法训练自编码器神经网络
	public void GetTargetNetwork(DoubleMatrix dataset, SimpleNetwork network,
			double max_error, int time, double learning_rate, double lamda,
			String bestPath) {
		DoubleMatrix sample = null;
		SimpleBackPropagation sbp = new SimpleBackPropagation(network);
		NetworkUpdateParameters ups = new NetworkUpdateParameters(network);
		double min_error = Double.MAX_VALUE;
		ups.zeroAll();
		for (int m = 0; m < time; m++) {
			for (int i = 0; i < dataset.columns; i++) {
				sample = dataset.getColumn(i);
				sbp.updateMatrixAndBias(sample, sample);
				ups.addAll(sbp.weights_updates, sbp.biass_updates);
			}
			ups.div(dataset.columns);
			ups.addLamdaWeights(lamda, network.weights);
			network.updateNet(ups.wus, ups.bus, learning_rate);
			// 计算误差
			double error = 0.0;
			_2LEvaluate evaluate = new _2LEvaluate();
			for (int i = 0; i < dataset.columns; i++) {
				sample = dataset.getColumn(i);
				error += evaluate.getError(sample, network.getOutput(sample));
			}
			error = error / dataset.columns;
			System.out.println((m + 1) + " : " + error);
			if (error < min_error) {
				SimpleNetwork.saveNetwork(bestPath, network);
				min_error = error;
			}
			if (error < max_error) {
				break;
			}
		}
		System.out.println(min_error);
	}

	// 根据样本使用CD-K算法初始化RBM
	public void GetTargetRBM(DoubleMatrix dataset, RBMNetwork rbm,
			double max_error, int time, double learning_rate, int cd_k,
			String bestPath) {
		DoubleMatrix sample = null;
		CDKBackPropagation cdkBP = new CDKBackPropagation(rbm);
		cdkBP.setK(cd_k);
		double min_error = Double.MAX_VALUE;
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
			error = error / dataset.columns;
			System.out.println((m + 1) + " : " + error);
			if (error < min_error) {
				RBMNetwork.saveNetwork(bestPath, rbm);
				min_error = error;
			}
			if (error < max_error) {
				break;
			}
		}
		System.out.println(min_error);
	}

	// 使用RBM生成第一层网络
	// @Test
	public void RBMGenerateL1() throws FileNotFoundException,
			ClassNotFoundException, IOException {
		int idx = 100;
		DoubleMatrix dataset = ReadDataset("/Users/terry/Desktop/dts_sub.dat");
		RBMNetwork rbm = new RBMNetwork(25, 15, 100);
		GetTargetRBM(dataset, rbm, 0.3, 50, 1.2, 1, "best/ts_l1_best.nt");
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

	// 使用RBM生成第二层网络结构
	// @Test
	public void RBMGenerateL2() throws FileNotFoundException,
			ClassNotFoundException, IOException {
		int idx = 100;
		DoubleMatrix dataset = ReadDataset("/Users/terry/Desktop/l1_feature_sub.dat");
		RBMNetwork rbm = new RBMNetwork(15, 10, 100);
		GetTargetRBM(dataset, rbm, 0.3, 50, 1.2, 1, "best/ts_l2_best.nt");
		SimpleNetwork firstNetwork = rbm.getNetwork();
		// 存储第一层特征提取层网络
		SimpleNetwork.saveNetwork("rbm/ts_l2", firstNetwork);
		SimpleNetwork secondNetwork = rbm.getRNetwork();
		firstNetwork.addUpperNetwork(secondNetwork);
		System.out.println(dataset.getColumn(idx));
		System.out.println(firstNetwork.getOutput(dataset.getColumn(idx)));
		// 存储特征提取层网络以及信号恢复层网络
		SimpleNetwork.saveNetwork("rbm/ts_l2r.nt", firstNetwork);
	}

	// @Test
	public void L2Training() throws FileNotFoundException,
			ClassNotFoundException, IOException {
		DoubleMatrix dataset = ReadDataset("/Users/terry/Desktop/l1_feature_sub.dat");
		SimpleNetwork ts_l1r = SimpleNetwork.loadNetwork("network/ts_l2r_2.nt");
		GetTargetNetwork(dataset, ts_l1r, 0.01, 30000, 1.8, 0.5,
				"best/ts_l2r_best.nt");
		SimpleNetwork.saveNetwork("network/ts_l2r_1.nt", ts_l1r);
	}

	// @Test
	public void GenreateBestL2InitNetwork() {
		RBMNetwork rbm = RBMNetwork.loadNetwork("best/ts_l2_best.nt");
		SimpleNetwork firstNetwork = rbm.getNetwork();
		SimpleNetwork secondNetwork = rbm.getRNetwork();
		firstNetwork.addUpperNetwork(secondNetwork);
		SimpleNetwork.saveNetwork("network/ts_l2r.nt", firstNetwork);
	}

	// @Test
	public void GenerateBestL1InitNetwork() {
		RBMNetwork rbm = RBMNetwork.loadNetwork("best/ts_l1_best.nt");
		SimpleNetwork firstNetwork = rbm.getNetwork();
		SimpleNetwork secondNetwork = rbm.getRNetwork();
		firstNetwork.addUpperNetwork(secondNetwork);
		SimpleNetwork.saveNetwork("network/ts_l1r.nt", firstNetwork);
	}

	// @Test
	public void L12Trainning() throws FileNotFoundException,
			ClassNotFoundException, IOException {
		DoubleMatrix dataset = ReadDataset("/Users/terry/Desktop/dts_sub.dat");
		SimpleNetwork ts_l1r = SimpleNetwork
				.loadNetwork("network/ts_l12r_1.nt");
		GetTargetNetwork(dataset, ts_l1r, 0.05, 5000, 0.8, 0.5,
				"best/ts_l1r_best.nt");
		SimpleNetwork.saveNetwork("network/ts_l12r_2.nt", ts_l1r);
	}

	// 载入RBM第一层模型使用BP
	// @Test
	public void L1Training() throws FileNotFoundException,
			ClassNotFoundException, IOException {
		DoubleMatrix dataset = ReadDataset("/Users/terry/Desktop/dts_sub.dat");
		SimpleNetwork ts_l1r = SimpleNetwork.loadNetwork("best/ts_l1r_best.nt");
		GetTargetNetwork(dataset, ts_l1r, 0.05, 100, 0.8, 0.5,
				"best/ts_l1r_best.nt");
		SimpleNetwork.saveNetwork("network/ts_l1r_1.nt", ts_l1r);
	}

	// 生成第一层特征
	// @Test
	public void Generate1LFeature() throws FileNotFoundException,
			ClassNotFoundException, IOException {
		SimpleNetwork ts_l1r = SimpleNetwork.loadNetwork("result/25_15_25.nt");
		DoubleMatrix dataset = ReadDataset("/Users/terry/Desktop/dts_all.dat");
		LinkedList<DoubleMatrix> features = ts_l1r.getOutputs(dataset);
		System.out.println(features.get(1).rows);
		SaveDataset("/Users/terry/Desktop/l1_feature_all.dat", features.get(1));
	}

	//
	// @Test
	public void test() throws FileNotFoundException, ClassNotFoundException,
			IOException {
		DoubleMatrix dataset = ReadDataset("/Users/terry/Desktop/l1_feature_all.dat");
		Random rand = new Random();
		for (int i = 0; i < 10; i++) {
			int idx = Math.abs(rand.nextInt()) % (274 * 214);
			System.out.println(idx + ":" + dataset.getColumn(idx));
		}
	}

	// 查看信号恢复情况
	@Test
	public void See1LRestoreSign() throws FileNotFoundException,
			ClassNotFoundException, IOException {
		DoubleMatrix dataset = ReadDataset("/Users/terry/Desktop/dts_all.dat");
		SimpleNetwork ts_l1r = SimpleNetwork
				.loadNetwork("network/noRBML1_1.nt");
		int idx = 24340;
		Random rand = new Random();
		for (int i = 0; i < 10; i++) {
			idx = Math.abs(rand.nextInt()) % (214 * 274);
			System.out.println(dataset.getColumn(idx).mul(12000).add(2000)
					.mul(0.0001));
			System.out.println(ts_l1r.getOutput(dataset.getColumn(idx))
					.mul(12000).add(2000).mul(0.0001));
		}
		// System.out.println(ts_l1r.getOutputs(dataset.getColumn(idx)).get(1));
	}

	// @Test
	public void See2LRestoreSign() throws FileNotFoundException,
			ClassNotFoundException, IOException {
		DoubleMatrix dataset = ReadDataset("/Users/terry/Desktop/l1_feature_all.dat");
		SimpleNetwork ts_l1r = SimpleNetwork.loadNetwork("network/ts_l2r_2.nt");
		int idx = 24340;
		Random rand = new Random();
		for (int i = 0; i < 1; i++) {
			idx = Math.abs(rand.nextInt()) % (214 * 274);
			// System.out.println(idx);
			System.out.println(dataset.getColumn(idx));
			// System.out.println(ts_l1r.getOutputs(dataset.getColumn(idx)).get(1));
			System.out.println(ts_l1r.getOutput(dataset.getColumn(idx)));
		}
	}

	// @Test
	public void SeeRestoreSign() throws FileNotFoundException,
			ClassNotFoundException, IOException {
		DoubleMatrix dataset = ReadDataset("/Users/terry/Desktop/dts_all.dat");
		SimpleNetwork ts_l1r = SimpleNetwork.loadNetwork("best/ts_l1r_best.nt");
		Random rand = new Random();
		for (int i = 0; i < 10; i++) {
			int idx = Math.abs(rand.nextInt()) % (214 * 274);
			System.out.println(dataset.getColumn(idx).mul(12000).add(2000)
					.mul(0.0001));
			System.out.println(ts_l1r.getOutput(dataset.getColumn(idx))
					.mul(12000).add(2000).mul(0.0001));
		}
	}

	// 生成特征
	// @Test
	public void GenerateFeature() throws FileNotFoundException,
			ClassNotFoundException, IOException {
		DoubleMatrix dataset = ReadDataset("/Users/terry/Desktop/dts_all.dat");
		SimpleNetwork ts_l1r = SimpleNetwork.loadNetwork("best/ts_l1r_best.nt");
		Random rand = new Random();
		LinkedList<DoubleMatrix> result = ts_l1r.getOutputs(dataset);
		for (int i = 50; i < 100; i++) {
			int idx = Math.abs(rand.nextInt()) % (214 * 274);
			System.out.println(idx % 274 + "," + (int) (idx / 274) + ":\n"
					+ result.get(2).getColumn(idx));
		}
		SaveDataset("/Users/terry/Desktop/ts_ features.dat", result.get(2));
	}

	// @Test
	public void NoRBMTrainningL1() throws FileNotFoundException,
			ClassNotFoundException, IOException {
		// SimpleNetwork network = new SimpleNetwork();
		// network.addLayer(new BasicLayer(25, false, new SigmoidFunction()));
		// network.addLayer(new BasicLayer(15, true, new SigmoidFunction()));
		// network.addLayer(new BasicLayer(25, true, new SigmoidFunction()));
		// network.initNetwork(100);
		SimpleNetwork network = SimpleNetwork
				.loadNetwork("network/noRBML1_1.nt");
		DoubleMatrix dataset = ReadDataset("/Users/terry/Desktop/dts_sub.dat");
		GetTargetNetwork(dataset, network, 0.05, 30000, 0.8, 0.5,
				"best/noRBML1.nt");
		SimpleNetwork.saveNetwork("network/noRBML1_2.nt", network);
	}

	// 使用生成的特征K-MEAN分类
	// @Test
	// public void Kmean() throws FileNotFoundException, ClassNotFoundException,
	// IOException {
	// }

	// @Test
	public void ConnectDeepNetwork() throws FileNotFoundException,
			ClassNotFoundException, IOException {
		SimpleNetwork n1 = SimpleNetwork.loadNetwork("result/25_15_25.nt");
		SimpleNetwork n2 = SimpleNetwork.loadNetwork("result/15_10_15.nt");
		DoubleMatrix dataset = ReadDataset("/Users/terry/Desktop/dts_all.dat");
		SimpleNetwork n = new SimpleNetwork();
		// 第一层网络输入
		// 25
		n.layers.add(n1.layers.get(0));
		// 添加第二层
		// 15
		n.layers.add(n1.layers.get(1));
		n.weights.add(n1.weights.get(0));
		n.biass.add(n1.biass.get(0));
		// 添加第三层
		// 10
		n.layers.add(n2.layers.get(1));
		n.weights.add(n2.weights.get(0));
		n.biass.add(n2.biass.get(0));
		// 添加第四层
		// 15
		n.layers.add(n2.layers.get(2));
		n.weights.add(n2.weights.get(1));
		n.biass.add(n2.biass.get(1));
		// 添加第五层
		// 25
		n.layers.add(n1.layers.get(2));
		n.weights.add(n1.weights.get(1));
		n.biass.add(n1.biass.get(1));
		SimpleNetwork.saveNetwork("network/ts_12_r.nt", n);
		int idx = 1024;
		System.out.println(dataset.getColumn(idx));
		System.out.println(n.getOutput(dataset.getColumn(idx)));
	}
}
