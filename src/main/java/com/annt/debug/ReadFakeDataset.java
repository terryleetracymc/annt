package com.annt.debug;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import com.annt.obj.DoubleSample;

public class ReadFakeDataset {

	static String HDFS_URL = "hdfs://192.168.1.140:9000/user/terry/";

	@SuppressWarnings("resource")
	public static void main(String[] args) {
		SparkConf conf = new SparkConf();
		conf.setAppName("readFakeDataset");
		conf.setMaster("local[4]");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		JavaRDD<DoubleSample> sampleRDD = jsc.objectFile(HDFS_URL+"fakeDataset");
		System.out.println(sampleRDD.count());
		jsc.stop();
	}

}
