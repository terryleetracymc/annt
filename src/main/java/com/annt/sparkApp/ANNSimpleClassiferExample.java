package com.annt.sparkApp;

import java.io.Serializable;
import java.util.Iterator;
import java.util.Random;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaRDDLike;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.broadcast.Broadcast;
import org.jblas.DoubleMatrix;

import scala.Tuple2;

import com.annt.function.SigmoidFunction;
import com.annt.layer.BasicLayer;
import com.annt.network.SimpleNetwork;
import com.annt.obj.DoubleSample;
import com.annt.obj.NetworkUpdate;
import com.annt.tranning.SimpleBackPropagation;
import com.obj.MNISTImageWithLabel;
import com.smiims.interf.sparkInput;

public class ANNSimpleClassiferExample implements sparkInput, Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 7682240049149068778L;
	private static final Random seed = new Random();
	static String SPARK_URL = "spark://192.168.1.140:7077";
	static String HDFS_URL = "hdfs://192.168.1.140:9000/user/terry/";
	public double lamda = 0.8;
	public double learning_rate = 1.5;

	static {
		// 设置日志等级
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
	}

	public static void main(String[] args) {
		ANNSimpleClassiferExample classifer = new ANNSimpleClassiferExample();
		JavaSparkContext jsc = classifer.getJSC();
		// 建立神经网络
		final Broadcast<SimpleNetwork> bNetwork = classifer.getNetWork(jsc);
		String[] argments = new String[] { HDFS_URL
				+ "MNISTDataset_sub/part-00000" };
		// 得到样本RDD
		JavaRDD<DoubleSample> sampleRDD = classifer.getData(jsc, argments);
		// 得到样本数量
		long sampleSize = sampleRDD.count();
		// 得到分组后数据
		JavaPairRDD<Integer, Iterable<DoubleSample>> groupedRDD = classifer
				.groupDataset(sampleRDD);
		// 缓存分组数据
		groupedRDD = groupedRDD.cache();
		double error = Double.MAX_VALUE;
		while (error > 0.1) {
			// 对分组后的数据进行训练
			JavaRDD<NetworkUpdate> updateRDD = classifer.train(bNetwork,
					groupedRDD);
			// 获得更新矩阵总和
			NetworkUpdate result = classifer.getUpdateSum(updateRDD);
			// 平均更新矩阵
			result.average(sampleSize);
			// 带权值衰减参数
			// 暂时不实现
			// 使用更新矩阵更新权值矩阵
			bNetwork.getValue().updateNet(result.matrix_updates,
					result.biass_updates, classifer.learning_rate);
			// 计算误差值
			DoubleMatrix errors = classifer.getErrors(bNetwork, groupedRDD);
			error = errors.divi(sampleSize).norm2();
			System.out.println(error);
		}
		jsc.stop();
	}

	// 平均一次权值更新矩阵
	public NetworkUpdate getUpdateSum(JavaRDD<NetworkUpdate> rdd) {
		return rdd
				.reduce(new Function2<NetworkUpdate, NetworkUpdate, NetworkUpdate>() {

					private static final long serialVersionUID = 8086728651393695556L;

					public NetworkUpdate call(NetworkUpdate v1, NetworkUpdate v2)
							throws Exception {
						v1.add(v2);
						return v1;
					}
				});
	}

	public DoubleMatrix getErrors(final Broadcast<SimpleNetwork> bNetwork,
			JavaPairRDD<Integer, Iterable<DoubleSample>> groudedRdd) {
		JavaRDD<DoubleMatrix> errorRDD = groudedRdd
				.map(new Function<Tuple2<Integer, Iterable<DoubleSample>>, DoubleMatrix>() {

					private static final long serialVersionUID = 7017834479083755743L;

					SimpleNetwork network = bNetwork.getValue();

					public DoubleMatrix call(
							Tuple2<Integer, Iterable<DoubleSample>> v)
							throws Exception {
						Iterator<DoubleSample> iterator = v._2.iterator();
						DoubleSample sample = iterator.next();
						DoubleMatrix errors = null;
						if (sample != null) {
							errors = DoubleMatrix.zeros(sample.ideal.rows);
							errors.addi(sample.ideal.subi(network
									.getOutput(sample.input)));
						}
						while (iterator.hasNext()) {
							sample = iterator.next();
							errors.addi(sample.ideal.subi(network
									.getOutput(sample.input)));
						}
						return errors;
					}
				});
		return errorRDD
				.reduce(new Function2<DoubleMatrix, DoubleMatrix, DoubleMatrix>() {

					/**
			 * 
			 */
					private static final long serialVersionUID = 1L;

					public DoubleMatrix call(DoubleMatrix v1, DoubleMatrix v2)
							throws Exception {
						return v1.add(v2);
					}
				});
	}

	// 训练
	public JavaRDD<NetworkUpdate> train(
			final Broadcast<SimpleNetwork> bNetwork,
			JavaPairRDD<Integer, Iterable<DoubleSample>> groupedRDD) {
		return groupedRDD
				.map(new Function<Tuple2<Integer, Iterable<DoubleSample>>, NetworkUpdate>() {
					private static final long serialVersionUID = -2819847373184226352L;

					SimpleNetwork network = bNetwork.getValue();

					public NetworkUpdate call(
							Tuple2<Integer, Iterable<DoubleSample>> v)
							throws Exception {
						//
						Iterator<DoubleSample> iterator = v._2.iterator();
						DoubleSample sample = iterator.next();
						NetworkUpdate nu = new NetworkUpdate();
						// 训练第一个样本
						SimpleBackPropagation sbp = new SimpleBackPropagation(
								network);
						if (sample != null) {
							sbp.getUpdateMatrixs(sample.input, sample.ideal);
							nu.addFirst(sbp.weights_updates, sbp.biass_updates);
						}
						while (iterator.hasNext()) {
							sample = iterator.next();
							sbp.getUpdateMatrixs(sample.input, sample.ideal);
							nu.add(sbp.weights_updates, sbp.biass_updates);
						}
						return nu;
					}
				});
	}

	// 将数据分组
	// 不分组会造成训练过程中
	// 更新矩阵网络传输数据过多
	public JavaPairRDD<Integer, Iterable<DoubleSample>> groupDataset(
			JavaRDD<DoubleSample> sampleRDD) {
		//
		JavaPairRDD<Integer, Iterable<DoubleSample>> groupedRDD = sampleRDD
				.groupBy(new Function<DoubleSample, Integer>() {
					private static final long serialVersionUID = 7125777744084913753L;

					public Integer call(DoubleSample v) throws Exception {
						return (Math.abs(v.hashCode() + seed.nextInt())) % 20;
					}
				});
		return groupedRDD;
	}

	// 得到JSC实例
	public JavaSparkContext getJSC() {
		SparkConf conf = new SparkConf();
		conf.set("spark.executor.memory", "5g");
		conf.set("spark.akka.frameSize", "5000");
		conf.set("spark.default.parallelism", "10");
		conf.set("spark.cores.max", "10");
		conf.set("spark.eventLog.enabled", "true");
		conf.set("spark.eventLog.dir", HDFS_URL + "eventlog/annt");
		conf.setJars(new String[] { HDFS_URL + "jars/sparkInterface.jar",
				HDFS_URL + "jars/mnist.jar", "/Users/terry/Desktop/annt.jar" });
		conf.setMaster(SPARK_URL);
		// conf.setMaster("local[4]");
		conf.setAppName(ANNSimpleClassiferExample.class.getName());
		JavaSparkContext jsc = new JavaSparkContext(conf);
		return jsc;
	}

	// 得到数据
	public JavaRDD<DoubleSample> getData(JavaSparkContext jsc, String[] argments) {
		if (argments == null || argments.length < 1) {
			return null;
		}
		// 读取原数据
		JavaRDD<MNISTImageWithLabel> rdd = jsc.objectFile(argments[0]);
		// 归一化原数据
		return rdd.map(new Function<MNISTImageWithLabel, DoubleSample>() {

			private static final long serialVersionUID = -5471264691303561502L;

			public DoubleSample call(MNISTImageWithLabel v) throws Exception {
				int label = v.label;
				int[] o_data = v.image;
				double[] data = new double[o_data.length];
				for (int i = 0; i < data.length; i++) {
					data[i] = o_data[i] / 255.0;
				}
				DoubleMatrix ideal = DoubleMatrix.zeros(10);
				ideal.put(label, 1.0);
				return new DoubleSample(new DoubleMatrix(data), ideal);
			}
		});
	}

	// 得到初始化的神经网络，广播数据
	public Broadcast<SimpleNetwork> getNetWork(JavaSparkContext jsc) {
		BasicLayer l1 = new BasicLayer(784, false, new SigmoidFunction());
		BasicLayer l2 = new BasicLayer(800, true, new SigmoidFunction());
		BasicLayer l3 = new BasicLayer(10, false, new SigmoidFunction());
		SimpleNetwork network = new SimpleNetwork();
		network.addLayer(l1);
		network.addLayer(l2);
		network.addLayer(l3);
		network.initNetwork(100);
		// 广播神经网络数据
		Broadcast<SimpleNetwork> b_network = jsc.broadcast(network);
		return b_network;
	}

	// spark接口输入输出
	@SuppressWarnings("rawtypes")
	public JavaRDDLike input(JavaSparkContext jsc, String[] input) {
		return null;
	}
}
