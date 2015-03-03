package com.annt.obj;

import java.io.Serializable;
import java.util.LinkedList;

import org.jblas.DoubleMatrix;

import com.annt.network.SimpleNetwork;

public class NetworkUpdateParameters implements Serializable {

	/**
	 * 神经网络更新参数
	 */
	private static final long serialVersionUID = 2087000805225177614L;

	// 权值更新
	public LinkedList<DoubleMatrix> wus;
	// 偏置更新
	public LinkedList<DoubleMatrix> bus;
	// 对应的神经网络
	public SimpleNetwork network;

	// 构造函数
	public NetworkUpdateParameters(SimpleNetwork nt) {
		network = nt;
		wus = new LinkedList<DoubleMatrix>();
		bus = new LinkedList<DoubleMatrix>();
		// 初始化权值更新
		for (int i = 0; i < nt.weights.size(); i++) {
			wus.add(DoubleMatrix.zeros(nt.weights.get(i).rows,
					nt.weights.get(i).columns));
		}
		// 初始化偏置更新
		for (int i = 0; i < nt.biass.size(); i++) {
			if (nt.biass.get(i) == null) {
				bus.add(null);
			} else {
				bus.add(DoubleMatrix.zeros(nt.biass.get(i).length));
			}
		}
	}

	// 添加权值衰减参数
	public void addLamdaWeights(double lamda, LinkedList<DoubleMatrix> wu) {
		for (int i = 0; i < wus.size(); i++) {
			wus.get(i).addi(wu.get(i).mul(lamda));
		}
	}

	public void zeroAll() {
		for (int i = 0; i < wus.size(); i++) {
			wus.get(i).subi(wus.get(i));
		}
		for (int i = 0; i < bus.size(); i++) {
			if (bus.get(i) != null) {
				bus.get(i).subi(bus.get(i));
			}
		}
	}

	public void div(int size) {
		for (int i = 0; i < wus.size(); i++) {
			wus.get(i).divi(size);
		}
		for (int i = 0; i < bus.size(); i++) {
			if (bus.get(i) != null) {
				bus.get(i).divi(size);
			}
		}
	}

	public void addWeights(LinkedList<DoubleMatrix> other) {
		for (int i = 0; i < wus.size(); i++) {
			wus.get(i).addi(other.get(i));
		}
	}

	public void addBiass(LinkedList<DoubleMatrix> other) {
		for (int i = 0; i < bus.size(); i++) {
			if (bus.get(i) != null) {
				bus.get(i).addi(other.get(i));
			}
		}
	}

	public void addAll(LinkedList<DoubleMatrix> ow, LinkedList<DoubleMatrix> ob) {
		addWeights(ow);
		addBiass(ob);
	}
}
