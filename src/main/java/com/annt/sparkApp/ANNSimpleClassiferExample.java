package com.annt.sparkApp;

import java.io.Serializable;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDDLike;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;

import com.annt.function.SigmoidFunction;
import com.annt.layer.BasicLayer;
import com.annt.network.SimpleNetwork;
import com.smiims.interf.sparkInput;

public class ANNSimpleClassiferExample implements sparkInput, Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 7682240049149068778L;
	static String SPARK_URL = "spark://192.168.1.140:7077";
	static String HDFS_URL = "hdfs://192.168.1.140:9000/user/terry/";

	public static void main(String[] args) {
		ANNSimpleClassiferExample classifer = new ANNSimpleClassiferExample();
		JavaSparkContext jsc = classifer.getJSC();
		// 建立神经网络
		final Broadcast<SimpleNetwork> bNetwork = classifer.getNetWork(jsc);
	}
	// 得到JSC实例
	public JavaSparkContext getJSC() {
		SparkConf conf = new SparkConf();
		conf.setAppName(ANNSimpleClassiferExample.class.getName());
		conf.set("spark.executor.memory", "5g");
		conf.set("spark.akka.frameSize", "5000");
		conf.set("spark.cores.max", "8");
		conf.set("spark.eventLog.enabled", "true");
		conf.set("spark.eventLog.dir", HDFS_URL + "eventlog/annt");
		conf.setJars(new String[] { HDFS_URL + "jars/sparkInterface.jar",
				HDFS_URL + "jars/mnist.jar", "/Users/liteng/Desktop/annt.jar" });
		conf.setMaster(SPARK_URL);
		// conf.setMaster("local[4]");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		return jsc;
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
	public JavaRDDLike input(JavaSparkContext arg0, String[] arg1) {
		return null;
	}
}
