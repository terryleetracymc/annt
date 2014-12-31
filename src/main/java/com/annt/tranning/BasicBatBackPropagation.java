package com.annt.tranning;

import java.util.LinkedList;

import org.jblas.DoubleMatrix;

import com.annt.network.SimpleNetwork;

abstract public class BasicBatBackPropagation extends SimpleBackPropagation {

	public BasicBatBackPropagation() {
	}

	public BasicBatBackPropagation(SimpleNetwork n) {
		super(n);
	}

	/**
	 * 批量样本输入的训练
	 */
	private static final long serialVersionUID = 7655722523220819291L;

	LinkedList<DoubleMatrix> w_sum_update;

	LinkedList<DoubleMatrix> b_sum_update;
}
