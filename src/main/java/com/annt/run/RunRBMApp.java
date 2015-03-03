package com.annt.run;

import java.io.Serializable;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaRDDLike;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.jblas.DoubleMatrix;

import com.annt.app.RBMApp;
import com.annt.network.RBMNetwork;
import com.annt.obj.RBMUpdateParameters;
import com.annt.obj.UnLabeledDoubleSample;
import com.annt.utils.CommonUtils;
import com.smiims.obj.GeoTSShortVector;

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

	public static JavaRDD<UnLabeledDoubleSample> prepareDataset(
			JavaSparkContext jsc) {
		JavaRDD<GeoTSShortVector> vectorsRDD = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/ts_data/MODIS/2000049_2010353/MOD13/h23v04/b1");
		// 数据采样
		vectorsRDD = vectorsRDD.sample(true, 0.00021);
		JavaRDD<UnLabeledDoubleSample> doubleVectorsRDD = vectorsRDD
				.map(new Function<GeoTSShortVector, UnLabeledDoubleSample>() {

					/**
					 * 将源数据转化为Double向量RDD
					 */
					private static final long serialVersionUID = -566735331026837033L;

					public UnLabeledDoubleSample call(GeoTSShortVector v)
							throws Exception {
						double data[] = new double[v.data.length];
						for (int i = 0; i < v.data.length; i++) {
							data[i] = v.data[i];
						}
						UnLabeledDoubleSample result = new UnLabeledDoubleSample(
								new DoubleMatrix(data));
						return result;
					}
				});
		// 数据归一化
		doubleVectorsRDD = doubleVectorsRDD
				.map(new Function<UnLabeledDoubleSample, UnLabeledDoubleSample>() {

					private static final long serialVersionUID = -4340324733631709708L;

					public UnLabeledDoubleSample call(UnLabeledDoubleSample v)
							throws Exception {
						v.input.addi(2000).divi(12000);
						return v;
					}
				});
		return doubleVectorsRDD;
	}

	public static void main(String args[]) {
		RunRBMApp app = new RunRBMApp("rbm_parameters.json");
		SparkConf conf = CommonUtils.readSparkConf("rbm_spark_conf.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		JavaRDD<UnLabeledDoubleSample> dataset = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/ts_data/annt_train/trainning_norm");
		dataset = dataset.sample(true, 0.05);
		dataset = dataset.cache();
		System.out.println(dataset.count());
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
		// 定义RBM网络
		RBMNetwork localRBMNetwork = new RBMNetwork(app.vDimension,
				app.hDimension, app.divRatio);
		final Broadcast<RBMNetwork> bRBMNetwork = jsc
				.broadcast(localRBMNetwork);
		double error = 0.0, min_error = Double.MAX_VALUE;// 一次训练
		for (int i = 0; i < app.time; i++) {
			RBMUpdateParameters updateParameters = app.train(bRBMNetwork,
					groupedDataset);
			// 更新参数求平均
			updateParameters.div(datasetSize);
			updateParameters.addLamdaWeight(0.05, localRBMNetwork.weight);
			localRBMNetwork.updateRBM(updateParameters.wu, updateParameters.vu,
					updateParameters.hu, app.learning_rate);
			DoubleMatrix errorVector = app
					.getError(bRBMNetwork, groupedDataset);
			error = errorVector.divi(datasetSize).norm2();
			if (min_error > error) {
				min_error = error;
				RBMNetwork.saveNetwork(app.rbmBestSavePath, localRBMNetwork);
			}
			System.out.println("第" + (i + 1) + "次迭代：" + error);
		}
		RBMNetwork.saveNetwork(app.rbmSavePath, localRBMNetwork);
		jsc.stop();
	}

	static {
		// 设置日志等级
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
	}

}
