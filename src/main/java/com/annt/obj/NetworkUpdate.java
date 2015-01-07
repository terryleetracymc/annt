package com.annt.obj;

import java.io.Serializable;
import java.util.LinkedList;

import org.jblas.DoubleMatrix;

public class NetworkUpdate implements Serializable {

	/**
	 * 用于神经矩阵和偏置更新的类
	 */
	private static final long serialVersionUID = 1951069583644637255L;

	// 表示是多少个矩阵的更新和
	public int numbers;
	// 矩阵更新
	public LinkedList<DoubleMatrix> matrix_updates;
	// 偏置更新
	public LinkedList<DoubleMatrix> biass_updates;
	// 是否已经是平均值
	public boolean isAverage;

	public NetworkUpdate(LinkedList<DoubleMatrix> mu,
			LinkedList<DoubleMatrix> bu, int n) {
		matrix_updates = mu;
		biass_updates = bu;
		numbers = n;
		isAverage = false;
	}

	//
	public static NetworkUpdate add(NetworkUpdate v1, NetworkUpdate v2) {
		NetworkUpdate result = new NetworkUpdate();
		return result;
	}

	// 平均化
	public void average() {
		for (int i = 0; i < matrix_updates.size(); i++) {
			matrix_updates.get(i).divi(numbers);
		}
		for (int i = 0; i < biass_updates.size(); i++) {
			if (biass_updates.get(i) != null) {
				biass_updates.get(i).divi(numbers);
			}
		}
		isAverage = true;
	}

	// 添加一个样本更新
	public void add(LinkedList<DoubleMatrix> mu, LinkedList<DoubleMatrix> bu) {
		for (int i = 0; i < matrix_updates.size(); i++) {
			matrix_updates.get(i).addi(mu.get(i));
		}
		for (int i = 0; i < biass_updates.size(); i++) {
			if (biass_updates.get(i) != null) {
				biass_updates.get(i).addi(bu.get(i));
			}
		}
		numbers++;
	}

	// 添加第一个样本
	public void addFirst(LinkedList<DoubleMatrix> mu,
			LinkedList<DoubleMatrix> bu) {
		matrix_updates = mu;
		biass_updates = bu;
		numbers = 1;
	}

	// 无参初始化
	public NetworkUpdate() {
		numbers = 0;
		matrix_updates = new LinkedList<DoubleMatrix>();
		biass_updates = new LinkedList<DoubleMatrix>();
	}
}
