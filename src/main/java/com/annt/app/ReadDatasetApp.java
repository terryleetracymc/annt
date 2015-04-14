package com.annt.app;

import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import com.annt.obj.UnLabeledDoubleSample;
import com.annt.utils.CommonUtils;

public class ReadDatasetApp {

	@SuppressWarnings("resource")
	public static void main(String[] args) {
		SparkConf conf = CommonUtils.readSparkConf("test_spark_conf.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		JavaRDD<UnLabeledDoubleSample> rdd = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/annt_train/trainning_norm/h26v05/b1/part-00000");
		List<UnLabeledDoubleSample> result = rdd.collect();
		rdd = rdd.sample(true, 0.1);
		for (int i = 0; i < result.size(); i++) {
			System.out.println(result.get(i).info);
		}
		jsc.stop();
	}

}
