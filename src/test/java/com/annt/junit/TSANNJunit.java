package com.annt.junit;

import java.io.Serializable;
import java.util.Random;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.junit.Test;

import com.annt.network.RBMNetwork;
import com.annt.network.SimpleNetwork;
import com.annt.utils.CommonUtils;

public class TSANNJunit implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6334123726225344004L;

	// @Test
	public void seeCenters() {
		DoubleMatrix centers = CommonUtils.ReadDataset("center1.mt");
		DoubleMatrix meanVector = DoubleMatrix.zeros(10);
		for (int i = 0; i < centers.columns; i++) {
			meanVector.addi(centers.getColumn(i));
		}
		meanVector.divi(20);
		DoubleMatrix erVector = DoubleMatrix.zeros(10);
		for (int i = 0; i < centers.columns; i++) {
			erVector.addi(MatrixFunctions.pow(centers.getColumn(i)
					.sub(erVector), 2));
		}
		System.out.println(erVector.norm2() / 10);
		centers = CommonUtils.ReadDataset("center2.mt");
		meanVector = DoubleMatrix.zeros(10);
		for (int i = 0; i < centers.columns; i++) {
			meanVector.addi(centers.getColumn(i));
		}
		meanVector.divi(20);
		erVector = DoubleMatrix.zeros(10);
		for (int i = 0; i < centers.columns; i++) {
			erVector.addi(MatrixFunctions.pow(centers.getColumn(i)
					.sub(erVector), 2));
		}
		System.out.println(erVector.norm2() / 10);
		centers = CommonUtils.ReadDataset("center3.mt");
		meanVector = DoubleMatrix.zeros(10);
		for (int i = 0; i < centers.columns; i++) {
			meanVector.addi(centers.getColumn(i));
		}
		meanVector.divi(20);
		erVector = DoubleMatrix.zeros(10);
		for (int i = 0; i < centers.columns; i++) {
			erVector.addi(MatrixFunctions.pow(centers.getColumn(i)
					.sub(erVector), 2));
		}
		System.out.println(erVector.norm2() / 10);
		centers = CommonUtils.ReadDataset("center4.mt");
		meanVector = DoubleMatrix.zeros(10);
		for (int i = 0; i < centers.columns; i++) {
			meanVector.addi(centers.getColumn(i));
		}
		meanVector.divi(20);
		erVector = DoubleMatrix.zeros(10);
		for (int i = 0; i < centers.columns; i++) {
			erVector.addi(MatrixFunctions.pow(centers.getColumn(i)
					.sub(erVector), 2));
		}
		System.out.println(erVector.norm2() / 10);
	}

	// @Test
	public void RBMTrain() {
		RBMNetwork rbm = RBMNetwork.loadNetwork("best/rbm_17_13.nt");
		SimpleNetwork firstNetwork = rbm.getNetwork();
		SimpleNetwork secondNetwork = rbm.getRNetwork();
		firstNetwork.addUpperNetwork(secondNetwork);
		SimpleNetwork.saveNetwork("best/17_13_17.nt", firstNetwork);
	}

	// @Test
	public void SeeNetwork() {
		SimpleNetwork network = SimpleNetwork.loadNetwork("best/23_17_23.nt");
		System.out.println(network.weights.getFirst());
	}

	@Test
	public void SeeRestoreSign() {
		SimpleNetwork network = SimpleNetwork.loadNetwork("best/23_17_23.nt");
		DoubleMatrix dataset = CommonUtils.ReadDataset("datasets/h27v06.mt");
		int time = 0;
		int iter_time = 100;
		for (int i = 0; i < iter_time; i++) {
			int idx = Math.abs(new Random().nextInt()) % iter_time;
			DoubleMatrix sample = dataset.getColumn(idx);
			DoubleMatrix output = network.getOutput(sample);
			DoubleMatrix error = MatrixFunctions.abs(sample.sub(output));
			DoubleMatrix feature = network.getOutputs(sample).get(1);
			double er = error.sub(error.mean()).norm2() / 23;
			if (er <= 0.004) {
				System.out.println("x1=" + sample.toString().replace(';', ','));
				System.out.println("x2=" + output.toString().replace(';', ','));
				System.out.println(feature);
				time++;
			} else {
			}
		}
		System.out.println((double) time / iter_time);
	}

	// @Test
	public void buildDeepNetwork() {
		SimpleNetwork n1 = SimpleNetwork.loadNetwork("best/23_17_13_10.nt");
		SimpleNetwork n2 = SimpleNetwork.loadNetwork("best/10_13_17_23.nt");
		n1.addUpperNetwork(n2);
		SimpleNetwork.saveNetwork("best/full.nt", n1);
		System.out.println(n1.getOutput(DoubleMatrix.rand(23)).length);
	}

	// @Test
	public void SeeFullRestoreSign() {
		SimpleNetwork full = SimpleNetwork.loadNetwork("best/full.nt");
		DoubleMatrix dataset = CommonUtils.ReadDataset("datasets/h25v05.mt");
		int iter_time = 20;
		for (int i = 0; i < iter_time; i++) {
			int idx = Math.abs(new Random().nextInt()) % 3000;
			DoubleMatrix sample = dataset.getColumn(idx);
			DoubleMatrix output = full.getOutput(sample);
			System.out.println("x1=" + sample.toString().replace(";", ","));
			System.out.println("x2=" + output.toString().replace(";", ","));
		}

	}

	// @Test
	public void rebuildDecoderNetwork() {
		SimpleNetwork l1 = SimpleNetwork.loadNetwork("best/23_17_23.nt");
		SimpleNetwork l2 = SimpleNetwork.loadNetwork("best/17_13_17.nt");
		SimpleNetwork l3 = SimpleNetwork.loadNetwork("best/13_10_13.nt");
		l1.removeFirstLayer();
		l2.removeFirstLayer();
		l3.removeFirstLayer();
		l2.addUpperNetwork(l1);
		l3.addUpperNetwork(l2);
		System.out.println(l3.getOutput(DoubleMatrix.rand(10)).length);
		SimpleNetwork.saveNetwork("best/10_13_17_23.nt", l3);
	}

	// @Test
	public void rebuildEncoderNetwork() {
		SimpleNetwork l1 = SimpleNetwork.loadNetwork("best/23_17_23.nt");
		SimpleNetwork l2 = SimpleNetwork.loadNetwork("best/17_13_17.nt");
		SimpleNetwork l3 = SimpleNetwork.loadNetwork("best/13_10_13.nt");
		// DoubleMatrix dataset = CommonUtils.ReadDataset("h27v06.mt");
		l1.removeLastLayer();
		l2.removeLastLayer();
		l3.removeLastLayer();
		l2.addLowwerNetwork(l1);
		l3.addLowwerNetwork(l2);
		SimpleNetwork.saveNetwork("best/23_17_13_10.nt", l3);
		// int iter_time = 20;
		// for (int i = 0; i < iter_time; i++) {
		// int idx = Math.abs(new Random().nextInt()) % 3000;
		// DoubleMatrix sample = dataset.getColumn(idx);
		// DoubleMatrix output = l3.getOutput(sample);
		// System.out.println(sample.toString().replace(";", ","));
		// System.out.println(output.toString().replace(";", ","));
		// }
	}

	// @Test
	public void ForDrawLine() {
		SimpleNetwork network = SimpleNetwork.loadNetwork("best/23_17_23.nt");
		DoubleMatrix dataset = CommonUtils.ReadDataset("test_sample_h27v06.mt");
		int iter_time = 20;
		for (int i = 0; i < iter_time; i++) {
			int idx = Math.abs(new Random().nextInt()) % 200000;
			DoubleMatrix sample = dataset.getColumn(idx);
			DoubleMatrix output = network.getOutput(sample);
			DoubleMatrix error = MatrixFunctions.abs(sample.sub(output));
			// double er = error.mean(); // error.sub(error.mean()).norm2() /
			// 23;
			System.out.println("y1=" + sample.toString().replace(';', ','));
			System.out.println("y2=" + output.toString().replace(';', ','));
			// System.out.println("f="
			// + network.getOutputs(sample).get(1).toString()
			// .replace(';', ','));
		}
		// System.out.println(error);
	}
}
