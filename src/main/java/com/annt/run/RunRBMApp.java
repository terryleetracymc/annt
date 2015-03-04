package com.annt.run;

import java.io.Serializable;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaRDDLike;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.jblas.DoubleMatrix;

import com.annt.app.RBMApp;
import com.annt.network.RBMNetwork;
import com.annt.obj.RBMUpdateParameters;
import com.annt.obj.UnLabeledDoubleSample;
import com.annt.utils.CommonUtils;

public class RunRBMApp implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3852835608027401829L;

	RBMApp app;

	//
	public RunRBMApp(String rbmJSONPath) {
		app = new RBMApp();
		app.loadConf(rbmJSONPath);
	}

	public static void main(String args[]) {
		RunRBMApp app = new RunRBMApp("rbm_parameters.json");
		SparkConf conf = CommonUtils.readSparkConf("rbm_spark_conf.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		JavaRDD<UnLabeledDoubleSample> dataset = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/ts_data/annt_train/trainning_norm");
		dataset = dataset.sample(true, 0.1);
		System.out.println(dataset.count());
		dataset = dataset.cache();
		app.run(jsc, dataset, null);
		jsc.stop();
	}

	@SuppressWarnings({ "rawtypes", "unchecked" })
	public void run(JavaSparkContext jsc, JavaRDDLike input, String[] args) {
		JavaRDD<UnLabeledDoubleSample> dataset = (JavaRDD<UnLabeledDoubleSample>) input;
		// 计算数据集的大小
		long datasetSize = dataset.count();
		JavaPairRDD<Integer, Iterable<UnLabeledDoubleSample>> groupedDataset = app
				.groupedDataset(dataset);
		groupedDataset = groupedDataset.cache();
		//
		final Broadcast<RBMNetwork> bRBMNetwork = jsc.broadcast(app.rbm);
		double error = 0.0, min_error = Double.MAX_VALUE;// 一次训练
		for (int i = 0; i < app.time; i++) {
			RBMUpdateParameters updateParameters = app.train(bRBMNetwork,
					groupedDataset);
			// 更新参数求平均
			updateParameters.div(datasetSize);
			updateParameters.addLamdaWeight(0.05, app.rbm.weight);
			app.rbm.updateRBM(updateParameters.wu, updateParameters.vu,
					updateParameters.hu, app.learning_rate);
			DoubleMatrix errorVector = app
					.getError(bRBMNetwork, groupedDataset);
			error = errorVector.divi(datasetSize).norm2();
			if (min_error > error) {
				min_error = error;
				RBMNetwork.saveNetwork(app.rbmBestSavePath, app.rbm);
			}
			System.out.println("第" + (i + 1) + "次迭代：" + error);
		}
		RBMNetwork.saveNetwork(app.rbmSavePath, app.rbm);
		jsc.stop();
	}

	static {
		// 设置日志等级
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
	}

}
