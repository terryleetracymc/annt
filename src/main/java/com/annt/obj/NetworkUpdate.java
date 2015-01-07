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

	public LinkedList<DoubleMatrix> matrix_updates;

	public LinkedList<DoubleMatrix> biass_updates;

	public NetworkUpdate(LinkedList<DoubleMatrix> mu,
			LinkedList<DoubleMatrix> bu, int n) {
		matrix_updates = mu;
		biass_updates = bu;
		numbers = n;
	}
}
