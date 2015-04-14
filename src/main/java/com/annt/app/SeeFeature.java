package com.annt.app;

import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import com.annt.obj.UnLabeledDoubleSample;
import com.annt.utils.CommonUtils;

public class SeeFeature {

	public static void main(String[] args) {
		SparkConf conf = CommonUtils.readSparkConf("default_spark.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		JavaRDD<UnLabeledDoubleSample> dataset = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/annt_train/trainning_norm/h27v06/selectSample2l/b1");
		dataset = dataset.sample(true, 0.00001);
		List<UnLabeledDoubleSample> result = dataset.collect();
		for (UnLabeledDoubleSample sample : result) {
			System.out.println(sample.data.length);
		}
		jsc.stop();
	}

}
