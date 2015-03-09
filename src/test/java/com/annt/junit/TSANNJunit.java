package com.annt.junit;

import java.io.Serializable;
import java.util.Random;

import org.jblas.DoubleMatrix;
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
	public void RBMTrain() {
		RBMNetwork rbm = RBMNetwork.loadNetwork("best/rbm_250_200.nt");
		SimpleNetwork firstNetwork = rbm.getNetwork();
		SimpleNetwork secondNetwork = rbm.getRNetwork();
		firstNetwork.addUpperNetwork(secondNetwork);
		SimpleNetwork.saveNetwork("network/250_200_250.nt", firstNetwork);
	}

	@Test
	public void SeeNetwork() {
		SimpleNetwork network = SimpleNetwork
				.loadNetwork("network/250_200_250_b2.nt");
		System.out.println(network.weights.getLast());
	}

	// @Test
	public void SeeRestoreSign() {
		SimpleNetwork network = SimpleNetwork
				.loadNetwork("network/250_200_250_b2.nt");
		DoubleMatrix dataset = CommonUtils.ReadDataset("test_sample.mt");
		int idx = Math.abs(new Random().nextInt()) % 3000;
		DoubleMatrix sample = dataset.getColumn(idx);
		System.out.println(sample.mul(12000).add(2000).mul(0.0001));
		System.out.println(network.getOutput(sample).mul(12000).add(2000)
				.mul(0.0001));
	}
}
