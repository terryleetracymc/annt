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

import com.annt.app.AutoEncoderApp;
import com.annt.network.SimpleNetwork;
import com.annt.obj.NetworkUpdateParameters;
import com.annt.obj.UnLabeledDoubleSample;
import com.annt.utils.CommonUtils;

public class RunLoadExistAutoEncoder implements Serializable {

	/**
	 * 神经网络并行训练
	 */
	private static final long serialVersionUID = -3846400550619484863L;

	AutoEncoderApp app;

	public static void main(String[] args) {
		RunLoadExistAutoEncoder app = new RunLoadExistAutoEncoder(
				"network/250_200_250.nt", "annt_parameters.json");
		SparkConf conf = CommonUtils.readSparkConf("annt_spark_conf.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		JavaRDD<UnLabeledDoubleSample> dataset = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/ts_data/annt_train/trainning_norm");
		dataset = dataset.sample(true, 0.1);
		dataset = dataset.cache();
		app.run(jsc, dataset, null);
		jsc.stop();
	}

	RunLoadExistAutoEncoder(String existANNPath, String confPath) {
		app = new AutoEncoderApp();
		app.loadExistANN(existANNPath);
		app.loadConf(confPath);
	}

	static {
		// 设置日志等级
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
		Logger.getLogger("apache").setLevel(Level.OFF);
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public void run(JavaSparkContext jsc, JavaRDDLike input, String[] args) {
		JavaRDD<UnLabeledDoubleSample> dataset = (JavaRDD<UnLabeledDoubleSample>) input;
		// 计算数据集的大小
		long datasetSize = dataset.count();
		JavaPairRDD<Integer, Iterable<UnLabeledDoubleSample>> groupedDataset = app
				.groupedDataset(dataset);
		groupedDataset = groupedDataset.cache();
		// 广播神经网络数据
		final Broadcast<SimpleNetwork> bNetwork = jsc.broadcast(app.network);
		double error = 0.0, min_error = Double.MAX_VALUE;
		for (int i = 0; i < app.time; i++) {
			NetworkUpdateParameters updateParameters = app.train(bNetwork,
					groupedDataset);
			// 更新参数求平均
			updateParameters.div(datasetSize);
			// 权值衰减参数
			updateParameters.addLamdaWeights(app.lamda, app.network.weights);
			app.network.updateNet(updateParameters.wus, updateParameters.bus,
					app.learning_rate);
			DoubleMatrix errorVector = app.getError(bNetwork, groupedDataset);
			error = errorVector.divi(datasetSize).norm2();
			if (min_error > error) {
				min_error = error;
				SimpleNetwork.saveNetwork(app.bestSavePath, app.network);
			}
			System.out.println("第" + (i + 1) + "次迭代：" + error);
		}
		SimpleNetwork.saveNetwork(app.savePath, app.network);
	}

}
