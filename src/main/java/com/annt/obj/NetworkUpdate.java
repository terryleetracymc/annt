package com.annt.obj;

import java.io.Serializable;
import java.util.LinkedList;

import org.jblas.DoubleMatrix;

public class NetworkUpdate implements Serializable {

	/**
	 * 用于神经矩阵和偏置更新的类
	 */
	private static final long serialVersionUID = 1951069583644637255L;

	// 矩阵更新
	public LinkedList<DoubleMatrix> weight_updates;
	// 偏置更新
	public LinkedList<DoubleMatrix> biass_updates;

	public NetworkUpdate(LinkedList<DoubleMatrix> mu,
			LinkedList<DoubleMatrix> bu) {
		weight_updates = mu;
		biass_updates = bu;
	}
	
	public void div(long num){
		for (int i = 0; i < weight_updates.size(); i++) {
			weight_updates.get(i).divi(num);
		}
		for (int i = 0; i < biass_updates.size(); i++) {
			if (biass_updates.get(i) != null) {
				biass_updates.get(i).divi(num);
			}
		}
	}

	// 相加操作
	public void add(NetworkUpdate v) {
		for (int i = 0; i < weight_updates.size(); i++) {
			weight_updates.get(i).addi(v.weight_updates.get(i));
		}
		for (int i = 0; i < biass_updates.size(); i++) {
			if (biass_updates.get(i) != null) {
				biass_updates.get(i).addi(v.biass_updates.get(i));
			}
		}
	}

	// 添加一个样本更新
	public void add(LinkedList<DoubleMatrix> mu, LinkedList<DoubleMatrix> bu) {
		for (int i = 0; i < weight_updates.size(); i++) {
			weight_updates.get(i).addi(mu.get(i));
		}
		for (int i = 0; i < biass_updates.size(); i++) {
			if (biass_updates.get(i) != null) {
				biass_updates.get(i).addi(bu.get(i));
			}
		}
	}

	// 添加第一个样本
	public void addFirst(LinkedList<DoubleMatrix> mu,
			LinkedList<DoubleMatrix> bu) {
		weight_updates = mu;
		biass_updates = bu;
	}

	// 无参初始化
	public NetworkUpdate() {
		weight_updates = new LinkedList<DoubleMatrix>();
		biass_updates = new LinkedList<DoubleMatrix>();
	}
}
