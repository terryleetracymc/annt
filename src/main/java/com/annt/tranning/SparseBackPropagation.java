package com.annt.tranning;

import org.jblas.DoubleMatrix;

import com.annt.network.SimpleNetwork;

public class SparseBackPropagation extends BasicBackPropagation {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7925806271170632395L;

	@Override
	void updateMatrixAndBias(DoubleMatrix input, DoubleMatrix ideal) {
	}

	public SparseBackPropagation(SimpleNetwork n) {
		network = n;
		weights = n.weights;
		biass = n.biass;
		layers = n.layers;
	}
}
