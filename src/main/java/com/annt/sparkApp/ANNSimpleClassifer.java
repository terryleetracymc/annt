package com.annt.sparkApp;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.util.Iterator;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.broadcast.Broadcast;
import org.jblas.DoubleMatrix;

import scala.Tuple2;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONException;
import com.alibaba.fastjson.JSONObject;
import com.annt.interf.ActiveFunction;
import com.annt.layer.BasicLayer;
import com.annt.network.SimpleNetwork;
import com.annt.obj.DoubleSample;
import com.annt.obj.NetworkUpdate;
import com.annt.tranning.SimpleBackPropagation;

public class ANNSimpleClassifer implements Serializable {
	/**
	 * spark_annt.json文件用于配置神经网络 输入为JavaRDD<DoubleSample>
	 */
	private static final long serialVersionUID = -7418183431014127688L;
	// 随机数种子
	private static Random seed = new Random();
	// SPARK集群url地址
	public String SPARK_URL;
	// 权值衰减参数
	public double lamda;
	// 学习率
	public double learning_rate;
	// spark conf对象
	public static SparkConf conf;
	// spark context对象
	public static JavaSparkContext jsc;
	// 神经网络对象
	public SimpleNetwork network;
	// 最大误差
	public double max_error;
	// 最大训练次数
	public int max_time;

	static {
		// 设置日志等级
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
	}

	public static void main(String[] args) throws FileNotFoundException {
		String inputPath = "/Users/terry/Desktop/fakeDataset";
		ANNSimpleClassifer classifer = new ANNSimpleClassifer();
		classifer.loadConf("spark_annt_local.json");
		// 读取原数据
		JavaRDD<DoubleSample> oRDD = classifer.readDataset(inputPath);
		// 为数据分组
		long dataset_size = oRDD.count();
		JavaPairRDD<Integer, Iterable<DoubleSample>> groupedRDD = classifer
				.groupDataset(oRDD);
		groupedRDD.cache();
		// 训练前将神经网络数据广播
		final Broadcast<SimpleNetwork> bNetwork = jsc
				.broadcast(classifer.network);
		double error = Double.MAX_VALUE;
		int time = 0;
		// ////********在此迭代********\\\\\\
		// 训练
		while (error > classifer.max_error) {
			JavaRDD<NetworkUpdate> updateRDD = classifer.train(bNetwork,
					groupedRDD);
			// 总权值更新
			NetworkUpdate nu = classifer.getUpdateSum(updateRDD);
			// 平均
			nu.div(dataset_size);
			// 使用权值更新更新神经网络
			bNetwork.getValue().updateNet(nu.weight_updates, nu.biass_updates,
					classifer.learning_rate);
			// 计算误差
			DoubleMatrix errorVector = classifer.getError(bNetwork, groupedRDD);
			error = errorVector.div(dataset_size).norm2();
			time++;
			// 超过迭代次数
			if (time > classifer.max_time) {
				break;
			}
		}
		jsc.stop();
	}

	public DoubleMatrix getError(final Broadcast<SimpleNetwork> bNetwork,
			JavaPairRDD<Integer, Iterable<DoubleSample>> groupRDD) {
		JavaRDD<DoubleMatrix> errorRDD = groupRDD
				.map(new Function<Tuple2<Integer, Iterable<DoubleSample>>, DoubleMatrix>() {
					private static final long serialVersionUID = 4411491857783070509L;

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
					private static final long serialVersionUID = -840656743025357422L;

					public DoubleMatrix call(DoubleMatrix v1, DoubleMatrix v2)
							throws Exception {
						return v1.add(v2);
					}
				});
	}

	// 聚合权值和偏置数值更新
	public NetworkUpdate getUpdateSum(JavaRDD<NetworkUpdate> updateRDD) {
		return updateRDD
				.reduce(new Function2<NetworkUpdate, NetworkUpdate, NetworkUpdate>() {
					private static final long serialVersionUID = -8455221601784861882L;

					public NetworkUpdate call(NetworkUpdate v1, NetworkUpdate v2)
							throws Exception {
						v1.add(v2);
						return v1;
					}
				});
	}

	// 训练函数
	public JavaRDD<NetworkUpdate> train(
			final Broadcast<SimpleNetwork> bNetwork,
			JavaPairRDD<Integer, Iterable<DoubleSample>> groupDataset) {
		return groupDataset
				.map(new Function<Tuple2<Integer, Iterable<DoubleSample>>, NetworkUpdate>() {

					private static final long serialVersionUID = 4991066614187033322L;

					SimpleNetwork nwk = bNetwork.getValue();

					public NetworkUpdate call(
							Tuple2<Integer, Iterable<DoubleSample>> v)
							throws Exception {
						SimpleBackPropagation sbp = new SimpleBackPropagation(
								nwk);
						Iterator<DoubleSample> iterator = v._2.iterator();
						NetworkUpdate nu = new NetworkUpdate();
						DoubleSample sample = iterator.next();
						sbp.getUpdateMatrixs(sample.input, sample.ideal);
						nu.addFirst(sbp.weights_updates, sbp.biass_updates);
						while (iterator.hasNext()) {
							sample = iterator.next();
							sbp.getUpdateMatrixs(sample.input, sample.ideal);
							nu.add(sbp.weights_updates, sbp.biass_updates);
						}
						return nu;
					}
				});
	}

	// 读取原始数据集
	public JavaRDD<DoubleSample> readDataset(String inputPath) {
		return jsc.objectFile(inputPath);
	}

	// 将数据集分组
	public JavaPairRDD<Integer, Iterable<DoubleSample>> groupDataset(
			JavaRDD<DoubleSample> dataset) {
		final Integer group_size = Integer.parseInt(conf
				.get("spark.default.parallelism"));
		return dataset.groupBy(new Function<DoubleSample, Integer>() {
			private static final long serialVersionUID = -8281625440711653868L;

			public Integer call(DoubleSample v) throws Exception {
				return Math.abs(v.hashCode() + seed.nextInt()) % group_size;
			}
		});
	}

	// 读取spark集群和神经网络配置
	@SuppressWarnings("resource")
	public boolean loadConf(String jsonPath) {
		String content;
		JSONObject conf_json;
		try {
			content = new Scanner(new File(jsonPath)).useDelimiter("\\Z")
					.next();
			conf_json = JSONObject.parseObject(content);
			// spark配置
			conf = new SparkConf();
			conf.setAppName(conf_json.getString("SPARK_APPNAME"));
			SPARK_URL = conf_json.getString("SPARK_URL");
			conf.setMaster(SPARK_URL);
			JSONObject spark_base_conf = conf_json
					.getJSONObject("SPARK_BASE_CONF");
			Set<String> keys = spark_base_conf.keySet();
			for (String key : keys) {
				conf.set(key, spark_base_conf.getString(key));
			}
			// 添加jar包
			JSONArray jars_array = conf_json.getJSONArray("jars");
			String[] jars_conf = new String[jars_array.size()];
			for (int i = 0; i < jars_conf.length; i++) {
				jars_conf[i] = jars_array.getString(i);
			}
			conf.setJars(jars_conf);
			jsc = new JavaSparkContext(conf);
			// 神经网络配置
			JSONObject annt_conf_json = conf_json.getJSONObject("ANN_CONF");
			JSONArray cell_array = annt_conf_json.getJSONArray("cell");
			JSONArray bias_array = annt_conf_json.getJSONArray("bias");
			JSONArray active_array = annt_conf_json.getJSONArray("active");
			double weight_ratio = annt_conf_json.getDouble("weight_ratio");
			network = new SimpleNetwork();
			for (int i = 0; i < cell_array.size(); i++) {
				// 反射获得激活函数
				Class<?> c = Class.forName(active_array.getString(i));
				ActiveFunction func = (ActiveFunction) c.newInstance();
				network.addLayer(new BasicLayer(cell_array.getIntValue(i),
						bias_array.getBooleanValue(i), func));
			}
			network.initNetwork(weight_ratio);
			// 学习率
			learning_rate = conf_json.getDoubleValue("learning_rate");
			// 权值衰减参数
			lamda = conf_json.getDoubleValue("lamda");
			// 最大误差
			max_error = conf_json.getDoubleValue("max_error");
			// 最多训练次数
			max_time = conf_json.getIntValue("max_time");
		} catch (FileNotFoundException e) {
			System.err.println("json配置文件不存在");
			return false;
		} catch (JSONException e) {
			System.err.println("JSON解析错误...");
			return false;
		} catch (ClassNotFoundException e) {
			System.err.println("未定义激活函数");
			return false;
		} catch (InstantiationException e) {
			System.err.println("激活函数无构造函数");
			return false;
		} catch (IllegalAccessException e) {
			System.err.println("激活函数权限出错");
			return false;
		}
		return true;
	}
}
