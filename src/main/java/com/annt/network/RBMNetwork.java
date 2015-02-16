package com.annt.network;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import org.jblas.DoubleMatrix;

import com.annt.function.SigmoidFunction;
import com.annt.layer.BasicLayer;

public class RBMNetwork extends BasicNetwork {

	/**
	 * 基于受限玻尔兹曼机的网络结构实现
	 */
	private static final long serialVersionUID = 2904054327850971298L;

	// 显层和隐层神经元个数
	public int vn, hn;
	// 连接显层和隐层的权值矩阵
	public DoubleMatrix weight;
	// 显层偏置
	public DoubleMatrix vbiass;
	// 隐层偏置
	public DoubleMatrix hbiass;
	// sigmoid函数
	public SigmoidFunction sigmoid = new SigmoidFunction();

	// 更新RBM网络参数
	public void updateRBM(DoubleMatrix wu, DoubleMatrix vbu, DoubleMatrix hbu,
			double learning_rate) {
		weight.addi(wu.mul(learning_rate));
		vbiass.addi(vbu.mul(learning_rate));
		hbiass.addi(hbu.mul(learning_rate));
	}

	// 将RBM网络结构返回定义的神经网络层
	public SimpleNetwork getNetwork() {
		SimpleNetwork network = new SimpleNetwork();
		network.addLayer(new BasicLayer(vn, false, new SigmoidFunction()));
		network.addLayer(new BasicLayer(hn, true, new SigmoidFunction()));
		network.weights.add(weight);
		network.biass.add(hbiass);
		return network;
	}

	// 返回RBM的反向神经网络结构
	public SimpleNetwork getRNetwork() {
		SimpleNetwork network = new SimpleNetwork();
		network.addLayer(new BasicLayer(hn, false, new SigmoidFunction()));
		network.addLayer(new BasicLayer(vn, true, new SigmoidFunction()));
		network.weights.add(weight.transpose());
		network.biass.add(vbiass);
		return network;
	}

	// 初始化RBM网络结构
	public RBMNetwork(int v, int h, int divRatio) {
		vn = v;
		hn = h;
		weight = DoubleMatrix.rand(vn, hn).div(divRatio);
		vbiass = DoubleMatrix.zeros(vn);
		hbiass = DoubleMatrix.zeros(hn);
	}

	// 得到隐藏层输出
	public DoubleMatrix getHOutput(DoubleMatrix input) {
		return sigmoid.output(weight, input, hbiass);
	}

	// 得到可视层输出
	public DoubleMatrix getVOutput(DoubleMatrix feature) {
		return sigmoid.output(weight.transpose(), feature, vbiass);
	}

	// 序列化存储神经网络
	public static void saveNetwork(String path, RBMNetwork network) {
		try {
			ObjectOutputStream out = new ObjectOutputStream(
					new FileOutputStream(path));
			out.writeObject(network);
			out.close();
		} catch (FileNotFoundException e) {
			// log
			e.printStackTrace();
		} catch (IOException e) {
			// log
			e.printStackTrace();
		}
	}

	// 载入神经网络
	public static RBMNetwork loadNetwork(String path) {
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(
					path));
			RBMNetwork network = (RBMNetwork) in.readObject();
			in.close();
			return network;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		return null;
	}
}
