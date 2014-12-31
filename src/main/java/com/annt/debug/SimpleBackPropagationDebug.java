package com.annt.debug;

import org.jblas.DoubleMatrix;

import com.annt.function.SigmoidFunction;
import com.annt.layer.BasicLayer;
import com.annt.network.SimpleNetwork;
import com.annt.tranning.SimpleBackPropagation;

public class SimpleBackPropagationDebug {

	public static void main(String[] args) {
		BasicLayer l1 = new BasicLayer(2, false, new SigmoidFunction());
		BasicLayer l2 = new BasicLayer(4, true, new SigmoidFunction());
		BasicLayer l3 = new BasicLayer(1, false, new SigmoidFunction());
		SimpleNetwork network = new SimpleNetwork();
		network.addLayer(l1);
		network.addLayer(l2);
		network.addLayer(l3);
		network.initNetwork(10);
		DoubleMatrix inputs[] = new DoubleMatrix[4];
		DoubleMatrix outputs[] = new DoubleMatrix[4];
		// 输入
		inputs[0] = new DoubleMatrix(new double[] { 0.0, 0.0 });
		inputs[1] = new DoubleMatrix(new double[] { 0.0, 1.0 });
		inputs[2] = new DoubleMatrix(new double[] { 1.0, 0.0 });
		inputs[3] = new DoubleMatrix(new double[] { 1.0, 1.0 });
		// 输出
		outputs[0] = new DoubleMatrix(new double[] { 0.0 });
		outputs[1] = new DoubleMatrix(new double[] { 1.0 });
		outputs[2] = new DoubleMatrix(new double[] { 1.0 });
		outputs[3] = new DoubleMatrix(new double[] { 0.0 });

		SimpleBackPropagation sbp = new SimpleBackPropagation(network);
		for (int m = 0; m < 2000; m++) {
			sbp.getUpdateMatrixs(inputs[0], outputs[0]);
			network.updateNet(sbp.weights_updates, sbp.biass_updates, 1);
			sbp.getUpdateMatrixs(inputs[1], outputs[1]);
			network.updateNet(sbp.weights_updates, sbp.biass_updates, 1);
			sbp.getUpdateMatrixs(inputs[2], outputs[2]);
			network.updateNet(sbp.weights_updates, sbp.biass_updates, 1);
			sbp.getUpdateMatrixs(inputs[3], outputs[3]);
			network.updateNet(sbp.weights_updates, sbp.biass_updates, 1);
		}
		System.out.println(network.getOutput(inputs[0]));
		System.out.println(network.getOutput(inputs[1]));
		System.out.println(network.getOutput(inputs[2]));
		System.out.println(network.getOutput(inputs[3]));
	}

}
