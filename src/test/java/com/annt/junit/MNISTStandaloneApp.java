package com.annt.junit;

import java.util.Random;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import com.annt.network.RBMNetwork;
import com.annt.network.SimpleNetwork;
import com.annt.utils.CommonUtils;

public class MNISTStandaloneApp {

	// @Test
	public void GenerateSubDataset() {
		DoubleMatrix dataset = CommonUtils
				.ReadDataset("/Users/terry/Desktop/mnist.dat");
		boolean isSelected[] = new boolean[60000];
		int size = 500;
		Random rand = new Random();
		DoubleMatrix sub_dataset = new DoubleMatrix(784, size);
		for (int i = 0; i < size; i++) {
			int idx = Math.abs(rand.nextInt()) % 60000;
			if (isSelected[idx]) {
				i--;
				continue;
			}
			isSelected[idx] = true;
			sub_dataset.putColumn(i, dataset.getColumn(idx).div(255));
		}
		CommonUtils.SaveDataset("/Users/terry/Desktop/sub_mnist.dat",
				sub_dataset);
	}

	@Test
	public void seeRestoreL1Sign() {
		DoubleMatrix dataset = CommonUtils
				.ReadDataset("/Users/terry/Desktop/sub_mnist.dat");
		SimpleNetwork network = SimpleNetwork
				.loadNetwork("network/mnist_rbm_l1r.nt");
		System.out.println(dataset.getColumn(0));
		System.out.println(network.getOutput(dataset.getColumn(0)));
	}

	// @Test
	public void L1Trainning() {
		SimpleNetwork network = SimpleNetwork
				.loadNetwork("network/mnist_l1r.nt");
		DoubleMatrix dataset = CommonUtils
				.ReadDataset("/Users/terry/Desktop/sub_mnist.dat");
		CommonUtils.GetTargetNetwork(dataset, network, 0.3, 300, 3, 0.5,
				"best/mnist_l1r_best.nt");
		SimpleNetwork.saveNetwork("network/mnist_l1r_1.nt", network);
	}

	// @Test
	public void GenerateL1() {
		RBMNetwork rbm = RBMNetwork.loadNetwork("best/mnist_rbm_l1r.nt");
		SimpleNetwork firstNetwork = rbm.getNetwork();
		SimpleNetwork secondNetwork = rbm.getRNetwork();
		firstNetwork.addUpperNetwork(secondNetwork);
		SimpleNetwork.saveNetwork("network/mnist_rbm_l1r.nt", firstNetwork);
	}

	// @Test
	public void GenerateRBMNetwork() {
		RBMNetwork rbm = new RBMNetwork(784, 600, 100);
		DoubleMatrix dataset = CommonUtils
				.ReadDataset("/Users/terry/Desktop/sub_mnist.dat");
		CommonUtils.GetTargetRBM(dataset, rbm, 0.5, 100, 3, 1,
				"best/mnist_rbm.nt");
		RBMNetwork.saveNetwork("rbm/mnist_rbm.nt", rbm);
	}
}
