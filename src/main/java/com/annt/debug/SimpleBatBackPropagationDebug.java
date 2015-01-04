package com.annt.debug;

import org.jblas.DoubleMatrix;

import com.annt.function.SigmoidFunction;
import com.annt.layer.BasicLayer;
import com.annt.network.SimpleNetwork;
import com.annt.tranning.SimpleBatBackPropagation;

public class SimpleBatBackPropagationDebug {

	public static void main(String[] args) {
		BasicLayer l1 = new BasicLayer(2, false, new SigmoidFunction());
		BasicLayer l2 = new BasicLayer(10, true, new SigmoidFunction());
		BasicLayer l3 = new BasicLayer(1, false, new SigmoidFunction());
		SimpleNetwork network = new SimpleNetwork();
		network.addLayer(l1);
		network.addLayer(l2);
		network.addLayer(l3);
		network.initNetwork(1);
		DoubleMatrix inputs = new DoubleMatrix(new double[][] {
				{ 0.2, 0.1, 1.0, 0.2 }, 
				{ 0.1, 0.5, 0.1, 1.0 } });
		DoubleMatrix outputs = new DoubleMatrix(new double[][] { { 0.02, 0.05,
				0.1, 0.2 } });
		SimpleBatBackPropagation sbbp = new SimpleBatBackPropagation(network);
		System.out.println(network.getOutput(inputs.getColumn(0)));
		System.out.println(network.getOutput(inputs.getColumn(1)));
		System.out.println(network.getOutput(inputs.getColumn(2)));
		System.out.println(network.getOutput(inputs.getColumn(3)));
		for (int m = 0; m < 10000; m++) {
			sbbp.updateMatrixAndBias(inputs, outputs);
			network.updateNet(sbbp.weights_updates, sbbp.biass_updates, 1.2);
		}
		System.out.println(network.getOutput(inputs.getColumn(0)));
		System.out.println(network.getOutput(inputs.getColumn(1)));
		System.out.println(network.getOutput(inputs.getColumn(2)));
		System.out.println(network.getOutput(inputs.getColumn(3)));
		System.out.println(network.getOutput(new DoubleMatrix(new double[]{0.3,0.6})));
	}
}
