package com.annt.network;

import java.util.LinkedList;

import org.jblas.DoubleMatrix;

import com.annt.interf.ActiveFunction;
import com.annt.layer.BasicLayer;

public class SimpleNetwork extends BasicNetwork {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8515281334911432901L;
	// 细胞层
	public LinkedList<BasicLayer> layers;
	// 权值矩阵
	public LinkedList<DoubleMatrix> weights;
	// 偏置
	public LinkedList<DoubleMatrix> biass;

	// 有些训练过程需要同时使用输出和激活值，不要调用getOutputs和getActives降低效率
	// 一次性调用getActivesAndOutputs
	// 结果返回到tmpOutputs和tmpActives中
	// 临时输出
	public LinkedList<DoubleMatrix> tmpOutputs;
	// 临时激活值
	public LinkedList<DoubleMatrix> tmpActives;

	// 构造函数
	public SimpleNetwork() {
		layers = new LinkedList<BasicLayer>();
		weights = new LinkedList<DoubleMatrix>();
		biass = new LinkedList<DoubleMatrix>();
	}

	// 添加神经层
	public void addLayer(BasicLayer l) {
		layers.add(l);
	}

	// 初始化神经网络
	public boolean initNetwork(double divRatio) {
		// 少于两层无法创建神经网络结构，divRatio等于0无法除
		if (layers.size() < 2 || divRatio == 0) {
			return false;
		}
		// 上一层网络
		BasicLayer lowwerLayer, upperLayer;
		// 每个权值矩阵输入输出维度
		int input_d, output_d;
		for (int i = 0; i < layers.size() - 1; i++) {
			lowwerLayer = layers.get(i);
			upperLayer = layers.get(i + 1);
			//
			input_d = lowwerLayer.neural_num;
			output_d = upperLayer.neural_num;
			// 两层神经网络确定一个权值矩阵
			DoubleMatrix w = DoubleMatrix.randn(input_d, output_d)
					.div(divRatio);
			weights.add(w);
			// 存在偏置矩阵
			if (upperLayer.bias) {
				DoubleMatrix b = DoubleMatrix.randn(output_d).div(divRatio);
				biass.add(b);
			} else {
				biass.add(null);
			}
		}
		return true;
	}

	public boolean initNetwork() {
		return initNetwork(1);
	}

	// 为某些训练过程设计的获得激活值和输出函数
	public void getActivesAndOutputs(DoubleMatrix input) {
		tmpActives = new LinkedList<DoubleMatrix>();
		tmpOutputs = new LinkedList<DoubleMatrix>();
		tmpActives.add(null);
		tmpOutputs.add(input);
		DoubleMatrix w;
		BasicLayer l;
		ActiveFunction f;
		DoubleMatrix x = input;
		DoubleMatrix b;
		for (int i = 0; i < weights.size(); i++) {
			w = weights.get(i);
			b = biass.get(i);
			l = layers.get(i + 1);
			f = l.activeFunc;
			x = f.active(w, x, b);
			tmpActives.add(x);
			x = f.output(x);
			tmpOutputs.add(x);
		}
	}

	public LinkedList<DoubleMatrix> getActives(DoubleMatrix input) {
		// 自上而下返回激活值
		LinkedList<DoubleMatrix> active = new LinkedList<DoubleMatrix>();
		// 添加第一层,激活输出第一层为null
		active.add(null);
		DoubleMatrix w;
		BasicLayer l;
		ActiveFunction f;
		DoubleMatrix x = input;
		DoubleMatrix b;
		for (int i = 0; i < weights.size(); i++) {
			w = weights.get(i);
			b = biass.get(i);
			l = layers.get(i + 1);
			f = l.activeFunc;
			x = f.active(w, x, b);
			active.add(x);
			x = f.output(x);
		}
		return active;
	}

	// 有些训练

	public LinkedList<DoubleMatrix> getOutputs(DoubleMatrix input) {
		// 自上而下返回输出值
		LinkedList<DoubleMatrix> output = new LinkedList<DoubleMatrix>();
		// 添加第一层,输出第一层为input
		output.add(input);
		DoubleMatrix w;
		BasicLayer l;
		ActiveFunction f;
		DoubleMatrix x = input;
		DoubleMatrix b;
		for (int i = 0; i < weights.size(); i++) {
			w = weights.get(i);
			b = biass.get(i);
			l = layers.get(i + 1);
			f = l.activeFunc;
			x = f.output(w, x, b);
			output.add(x);
		}
		return output;
	}

	public void updateNet(LinkedList<DoubleMatrix> w,
			LinkedList<DoubleMatrix> b, double learning_rate) {
		for (int i = 0; i < weights.size(); i++) {
			DoubleMatrix ow = weights.get(i);
			DoubleMatrix ob = biass.get(i);
			DoubleMatrix uw = w.get(i);
			DoubleMatrix ub = b.get(i);
			ow = ow.sub(uw.mul(learning_rate));
			if (ob != null) {
				ob.sub(ub.mul(learning_rate));
				biass.set(i, ob);
			}
			weights.set(i, ow);
		}
	}

	public DoubleMatrix getOutput(DoubleMatrix input) {
		LinkedList<DoubleMatrix> outputs = getOutputs(input);
		return outputs.getLast();
	}
}
