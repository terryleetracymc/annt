package com.annt.junit;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import com.alibaba.fastjson.JSONObject;
import com.annt.function.SigmoidFunction;
import com.annt.layer.BasicLayer;
import com.annt.network.RBMNetwork;
import com.annt.network.SimpleNetwork;
import com.annt.obj.PixelUnLabeledDoubleSample;
import com.annt.trainning.CDKBackPropagation;
import com.annt.trainning.SimpleBackPropagation;
import com.annt.utils.CommonUtils;
import com.smiims.obj.GeoTSShortVector;

public class ANNTest {

	// @Test
	public void SaveLoadTest() {
		BasicLayer l1 = new BasicLayer(2, false, new SigmoidFunction());
		BasicLayer l2 = new BasicLayer(4, true, new SigmoidFunction());
		BasicLayer l3 = new BasicLayer(1, false, new SigmoidFunction());
		SimpleNetwork network = new SimpleNetwork();
		network.addLayer(l1);
		network.addLayer(l2);
		network.addLayer(l3);
		network.initNetwork(100);
		System.out.println(network.weights.get(0));
		SimpleNetwork.saveNetwork("network.nt", network);
		SimpleNetwork anet = SimpleNetwork.loadNetwork("network.nt");
		System.out.println(anet.weights.get(0));
	}

	@SuppressWarnings("unchecked")
	@Test
	public void RBMTSTest() throws FileNotFoundException, IOException,
			ClassNotFoundException {
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				"/Users/terry/Desktop/dts.dat"));
		ArrayList<PixelUnLabeledDoubleSample> dataset=(ArrayList<PixelUnLabeledDoubleSample>) in.readObject();
		in.close();
	}

	// @Test
	public void OneBPTest() {
		BasicLayer l1 = new BasicLayer(2, false, new SigmoidFunction());
		BasicLayer l2 = new BasicLayer(4, true, new SigmoidFunction());
		BasicLayer l3 = new BasicLayer(1, false, new SigmoidFunction());
		SimpleNetwork network = new SimpleNetwork();
		network.addLayer(l1);
		network.addLayer(l2);
		network.addLayer(l3);
		network.initNetwork(1);
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
		for (int m = 0; m < 1000; m++) {
			sbp.updateMatrixAndBias(inputs[0], outputs[0]);
			network.updateNet(sbp.weights_updates, sbp.biass_updates, 1.8);
			sbp.updateMatrixAndBias(inputs[1], outputs[1]);
			network.updateNet(sbp.weights_updates, sbp.biass_updates, 1.8);
			sbp.updateMatrixAndBias(inputs[2], outputs[2]);
			network.updateNet(sbp.weights_updates, sbp.biass_updates, 1.8);
			sbp.updateMatrixAndBias(inputs[3], outputs[3]);
			network.updateNet(sbp.weights_updates, sbp.biass_updates, 1.8);
		}
		System.out.println(network.getOutput(inputs[0]));
		System.out.println(network.getOutput(inputs[1]));
		System.out.println(network.getOutput(inputs[2]));
		System.out.println(network.getOutput(inputs[3]));
	}

	// @Test
	public void RBMTest() {
		RBMNetwork rbm = new RBMNetwork(5, 3, 100);
		DoubleMatrix sample = DoubleMatrix.rand(5);
		CDKBackPropagation cdkBP = new CDKBackPropagation(rbm);
		cdkBP.setK(1);
		DoubleMatrix hOutput = null;
		hOutput = rbm.getHOutput(sample);
		System.out.println(rbm.getVOutput(hOutput));
		System.out.println(sample);
		for (int i = 0; i < 1000; i++) {
			cdkBP.updateMatrixAndBias(sample);
			rbm.updateRBM(cdkBP.wu, cdkBP.vbu, cdkBP.hbu, 1);
		}
		hOutput = rbm.getHOutput(sample);
		System.out.println(rbm.getVOutput(hOutput));
		System.out.println(rbm.weight);
	}

	// @Test
	public void ANNInitByJSON() throws FileNotFoundException {
		JSONObject json = JSONObject.parseObject(CommonUtils
				.readJSONText("annt.json"));
		SimpleNetwork network = new SimpleNetwork(json);
		System.out.println(network.weights.get(1));
	}

	// @Test
	public void loadDatsetTest() throws FileNotFoundException, IOException,
			ClassNotFoundException {
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				"/Users/terry/Desktop/ts.dat"));
		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(
				"/Users/terry/Desktop/dts.dat"));
		ArrayList<PixelUnLabeledDoubleSample> dataset = new ArrayList<PixelUnLabeledDoubleSample>();
		for (int i = 0; i < 274 * 214; i++) {
			GeoTSShortVector vector = (GeoTSShortVector) in.readObject();
			int x = vector.x;
			int y = vector.y;
			double d[] = new double[vector.data.length];
			for (int j = 0; j < d.length; j++) {
				d[j] = vector.data[j];
				d[j] = (d[j] + 2000) / 12000;
			}
			DoubleMatrix data = new DoubleMatrix(d);
			PixelUnLabeledDoubleSample sample = new PixelUnLabeledDoubleSample(
					data, x, y);
			dataset.add(sample);
		}
		out.writeObject(dataset);
		out.close();
		in.close();
	}

	// @Test
	public void ANNConnectTest() {
		SimpleNetwork n1 = new SimpleNetwork();
		n1.addLayer(new BasicLayer(10, false, new SigmoidFunction()));
		n1.addLayer(new BasicLayer(5, true, new SigmoidFunction()));
		n1.initNetwork();
		SimpleNetwork n2 = new SimpleNetwork();
		n2.addLayer(new BasicLayer(5, false, new SigmoidFunction()));
		n2.addLayer(new BasicLayer(10, true, new SigmoidFunction()));
		n2.initNetwork();
		boolean result = n2.addLowwerNetwork(n1);
		System.out.println(n2.layers.get(0).neural_num);
		System.out.println(n2.layers.get(1).neural_num);
		System.out.println(n2.layers.get(2).neural_num);
		System.out.println(n2.getOutput(DoubleMatrix.rand(10)));
		System.out.println(result);
	}
}
