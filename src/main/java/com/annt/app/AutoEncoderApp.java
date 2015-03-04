package com.annt.app;

import java.util.Iterator;
import java.util.Random;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.broadcast.Broadcast;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import scala.Tuple2;

import com.alibaba.fastjson.JSONObject;
import com.annt.network.SimpleNetwork;
import com.annt.obj.NetworkUpdateParameters;
import com.annt.obj.UnLabeledDoubleSample;
import com.annt.trainning.SimpleBackPropagation;
import com.annt.utils.CommonUtils;

public class AutoEncoderApp extends SparkApp {

	/**
	 * 自编码器SparkApp
	 */
	private static final long serialVersionUID = -9088122197801375730L;

	public int time;

	public double max_error;

	public double learning_rate;

	public double lamda;

	public int groupNum;

	public String savePath;

	public String bestSavePath;
	// 随机数种子
	public Random seed = new Random();

	public SimpleNetwork network;

	public AutoEncoderApp() {
	}

	// 读取RBM配置
	public void loadConf(String path) {
		JSONObject json_conf = JSONObject.parseObject(CommonUtils
				.readJSONText(path));
		// 读取各种RBM的参数
		savePath = json_conf.getString("savePath");
		bestSavePath = json_conf.getString("bestSavePath");
		time = json_conf.getIntValue("time");
		max_error = json_conf.getDoubleValue("max_error");
		learning_rate = json_conf.getDoubleValue("learning_rate");
		lamda = json_conf.getDoubleValue("lamda");
		groupNum = json_conf.getIntValue("groupNum");
	}

	// 载入神经网络，两种方式，一种自定义，一种是载入已有的神经网络
	// 载入已有的神经网络
	public void loadExistANN(String path) {
		network = SimpleNetwork.loadNetwork(path);
	}

	// 使用json文件初始化神经网络
	public void loadANNByJSON(String path) {
		network = new SimpleNetwork(JSONObject.parseObject(CommonUtils
				.readJSONText(path)));
	}

	// 将数据集分组
	public JavaPairRDD<Integer, Iterable<UnLabeledDoubleSample>> groupedDataset(
			JavaRDD<UnLabeledDoubleSample> dataset) {
		return dataset.groupBy(new Function<UnLabeledDoubleSample, Integer>() {

			private static final long serialVersionUID = 487890844179550294L;

			public Integer call(UnLabeledDoubleSample v) throws Exception {
				int id = Math.abs(v.hashCode() + seed.nextInt()) % groupNum;
				return id;
			}
		});
	}

	@SuppressWarnings("resource")
	public static void main(String args[]) {
		SparkConf conf = CommonUtils.readSparkConf("annt_spark_conf.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		AutoEncoderApp encoder = new AutoEncoderApp();
		encoder.loadConf("annt_parameters.json");
		encoder.loadExistANN("network/250_200_250.nt");

		JavaRDD<UnLabeledDoubleSample> dataset = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/ts_data/annt_train/trainning_norm");
		dataset = dataset.sample(true, 0.1);
		long datasetSize = dataset.count();

		JavaPairRDD<Integer, Iterable<UnLabeledDoubleSample>> groupedDataset = encoder
				.groupedDataset(dataset);
		groupedDataset.cache();

		final Broadcast<SimpleNetwork> bNetwork = jsc
				.broadcast(encoder.network);
		double error = Double.MAX_VALUE;

		for (int i = 0; i < 100; i++) {
			NetworkUpdateParameters ups = encoder.train(bNetwork,
					groupedDataset);
			ups.div(datasetSize);
			ups.addLamdaWeights(encoder.lamda, encoder.network.weights);

			encoder.network.updateNet(ups.wus, ups.bus, encoder.learning_rate);
			
			DoubleMatrix errorVector = encoder.getError(bNetwork,
					groupedDataset);

			error = errorVector.norm2();

			System.out.println(error / datasetSize);
		}

		jsc.stop();
	}

	// 分组训练
	public NetworkUpdateParameters train(
			final Broadcast<SimpleNetwork> network,
			JavaPairRDD<Integer, Iterable<UnLabeledDoubleSample>> groupedDataset) {
		JavaRDD<NetworkUpdateParameters> updateParameters = groupedDataset
				.map(new Function<Tuple2<Integer, Iterable<UnLabeledDoubleSample>>, NetworkUpdateParameters>() {

					private static final long serialVersionUID = -4206001993907080274L;
					SimpleNetwork net = network.getValue();

					public NetworkUpdateParameters call(
							Tuple2<Integer, Iterable<UnLabeledDoubleSample>> v)
							throws Exception {
						SimpleBackPropagation sbp = new SimpleBackPropagation(
								net);
						Iterator<UnLabeledDoubleSample> iterator = v._2
								.iterator();
						NetworkUpdateParameters ups = new NetworkUpdateParameters(
								net);
						while (iterator.hasNext()) {
							UnLabeledDoubleSample sample = iterator.next();
							sbp.updateMatrixAndBias(sample.data, sample.data);
							ups.addAll(sbp.weights_updates, sbp.biass_updates);
						}
						return ups;
					}
				});
		return updateParameters
				.reduce(new Function2<NetworkUpdateParameters, NetworkUpdateParameters, NetworkUpdateParameters>() {

					private static final long serialVersionUID = 3457258456922329907L;

					public NetworkUpdateParameters call(
							NetworkUpdateParameters v1,
							NetworkUpdateParameters v2) throws Exception {
						v1.addAll(v2.wus, v2.bus);
						return v1;
					}
				});
	}

	// 获得当前误差
	public DoubleMatrix getError(final Broadcast<SimpleNetwork> network,
			JavaPairRDD<Integer, Iterable<UnLabeledDoubleSample>> groupedDataset) {
		JavaRDD<DoubleMatrix> errors = groupedDataset
				.map(new Function<Tuple2<Integer, Iterable<UnLabeledDoubleSample>>, DoubleMatrix>() {

					private static final long serialVersionUID = 5884326774007366131L;

					SimpleNetwork net = network.getValue();

					public DoubleMatrix call(
							Tuple2<Integer, Iterable<UnLabeledDoubleSample>> v)
							throws Exception {
						DoubleMatrix error = DoubleMatrix.zeros(net.layers
								.getLast().neural_num);
						Iterator<UnLabeledDoubleSample> iterator = v._2
								.iterator();
						while (iterator.hasNext()) {
							UnLabeledDoubleSample sample = iterator.next();
							DoubleMatrix output = net.getOutput(sample.data);
							error.addi(MatrixFunctions.abs(output
									.sub(sample.data)));
						}
						return error;
					}
				});
		return errors
				.reduce(new Function2<DoubleMatrix, DoubleMatrix, DoubleMatrix>() {

					private static final long serialVersionUID = -472011257192204650L;

					public DoubleMatrix call(DoubleMatrix v1, DoubleMatrix v2)
							throws Exception {
						return v1.addi(v2);
					}
				});
	}
}
