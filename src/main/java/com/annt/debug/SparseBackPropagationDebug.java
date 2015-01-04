package com.annt.debug;

import java.util.LinkedList;

import org.jblas.DoubleMatrix;

import com.annt.function.SigmoidFunction;
import com.annt.layer.BasicLayer;
import com.annt.network.SimpleNetwork;
import com.annt.tranning.SparseBackPropagation;

public class SparseBackPropagationDebug {

	public static void main(String[] args) {
		BasicLayer l1 = new BasicLayer(2, false, new SigmoidFunction());
		BasicLayer l2 = new BasicLayer(5, true, new SigmoidFunction());
		BasicLayer l3 = new BasicLayer(1, false, new SigmoidFunction());
		LinkedList<Integer> sparse_num = new LinkedList<Integer>();
		sparse_num.add(0);
		sparse_num.add(2);
		sparse_num.add(0);
		SimpleNetwork network = new SimpleNetwork();
		network.addLayer(l1);
		network.addLayer(l2);
		network.addLayer(l3);
		network.initNetwork(10);
		SparseBackPropagation sbp = new SparseBackPropagation(network);
		// 设置稀疏参数
		LinkedList<DoubleMatrix> sparse_parameters = new LinkedList<DoubleMatrix>();
		sparse_parameters.add(DoubleMatrix.zeros(2));
		sparse_parameters.add(new DoubleMatrix(new double[] { 0.8, 0.8, 0.0,
				0.0, 0.0 }));
		sparse_parameters.add(DoubleMatrix.zeros(1));
		sbp.setSparse(sparse_parameters);
		// 设置抑制的输出值
		sbp.setExcepActive(0.8);
		DoubleMatrix inputs = new DoubleMatrix(new double[][] {
				{ 0.0, 0.0, 1.0, 1.0 }, { 0.0, 1.0, 0.0, 1.0 } });
		DoubleMatrix outputs = new DoubleMatrix(new double[][] { { 0.0, 1.0,
				1.0, 0.0 } });
		System.out.println(network.getActives(inputs.getColumn(1)).get(1));
		for (int m = 0; m < 2000; m++) {
			sbp.getUpdateMatrixs(inputs, outputs);
			network.updateNet(sbp.weights_updates, sbp.biass_updates, 0.8);
		}
		System.out.println(network.getActives(inputs.getColumn(1)).get(1));
	}

}
