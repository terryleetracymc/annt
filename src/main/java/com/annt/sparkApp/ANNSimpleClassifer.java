package com.annt.sparkApp;

import java.io.Serializable;

import org.apache.commons.math3.ml.neuralnet.Network;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaRDDLike;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.jblas.DoubleMatrix;

import com.annt.function.SigmoidFunction;
import com.annt.layer.BasicLayer;
import com.annt.network.SimpleNetwork;
import com.annt.obj.DoubleLabelSample;
import com.annt.obj.LabelSample;
import com.obj.MNISTImageWithLabel;
import com.smiims.interf.sparkInput;

public class ANNSimpleClassifer implements sparkInput, Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 7682240049149068778L;
	static String SPARK_URL = "spark://192.168.1.140:7077";
	static String HDFS_URL = "hdfs://192.168.1.140:9000/user/terry/";

	@SuppressWarnings("unchecked")
	public static void main(String[] args) {
		ANNSimpleClassifer classifer = new ANNSimpleClassifer();
		JavaSparkContext jsc = classifer.getJSC();
		// 输入参数
		String argments[] = new String[] { HDFS_URL + "MNISTDataset/" };
		// 训练数据集RDD
		JavaRDD<MNISTImageWithLabel> datasetRDD = (JavaRDD<MNISTImageWithLabel>) classifer
				.getData(jsc, argments);
		// 将数据集归一化
		JavaRDD<DoubleLabelSample> ndatasetRDD = (JavaRDD<DoubleLabelSample>) classifer
				.normalize(datasetRDD, null);
		// 得到神经网络
		final Broadcast<SimpleNetwork> network = classifer.getNetWork(jsc);
		jsc.stop();
	}

	// 得到JSC实例
	public JavaSparkContext getJSC() {
		SparkConf conf = new SparkConf();
		conf.setAppName("GenerateMNISTData");
		conf.set("spark.executor.memory", "3g");
		conf.set("spark.cores.max", "8");
		conf.setJars(new String[] { HDFS_URL + "jars/sparkInterface.jar",
				HDFS_URL + "jars/mnist.jar", "/Users/liteng/Desktop/annt.jar" });
		conf.setMaster(SPARK_URL);
		JavaSparkContext jsc = new JavaSparkContext(conf);
		return jsc;
	}

	// 得到初始化的神经网络
	public Broadcast<SimpleNetwork> getNetWork(JavaSparkContext jsc) {
		BasicLayer l1 = new BasicLayer(784, false, new SigmoidFunction());
		BasicLayer l2 = new BasicLayer(800, true, new SigmoidFunction());
		BasicLayer l3 = new BasicLayer(10, false, new SigmoidFunction());
		SimpleNetwork network = new SimpleNetwork();
		network.addLayer(l1);
		network.addLayer(l2);
		network.addLayer(l3);
		network.initNetwork(100);
		// 广播神经网络数据，广播数据为final
		Broadcast<SimpleNetwork> b_network = jsc.broadcast(network);
		return b_network;
	}

	@SuppressWarnings("rawtypes")
	public JavaRDDLike input(JavaSparkContext jsc, String[] args) {
		if (args.length < 1) {
			return null;
		}
		return null;
	}

	// 原数据归一化
	@SuppressWarnings({ "unchecked", "rawtypes" })
	public JavaRDDLike normalize(JavaRDDLike rdd, String[] args) {
		JavaRDD<MNISTImageWithLabel> datasetRDD = (JavaRDD<MNISTImageWithLabel>) rdd;
		JavaRDD<LabelSample> ndatasetRDD = datasetRDD
				.map(new Function<MNISTImageWithLabel, LabelSample>() {

					private static final long serialVersionUID = -7836842415386825988L;

					public LabelSample call(MNISTImageWithLabel v)
							throws Exception {
						int[] ori_data = v.image;
						double[] data = new double[ori_data.length];
						for (int i = 0; i < ori_data.length; i++) {
							data[i] = ori_data[i] / 255.0;
						}
						int label = v.label;
						return new DoubleLabelSample(new DoubleMatrix(data),
								label);
					}
				});
		return ndatasetRDD;
	}

	// 读取数据
	@SuppressWarnings("rawtypes")
	public JavaRDDLike getData(JavaSparkContext jsc, String[] args) {
		if (args.length < 1) {
			return null;
		}
		JavaRDD<MNISTImageWithLabel> rdd = jsc.objectFile(args[0]);
		return rdd;
	}

}
