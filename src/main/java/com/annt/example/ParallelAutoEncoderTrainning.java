package com.annt.example;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import com.annt.obj.UnLabeledDoubleSample;
import com.annt.run.RunLoadExistAutoEncoder;
import com.annt.utils.CommonUtils;

public class ParallelAutoEncoderTrainning {

	public static void main(String[] args) {
		RunLoadExistAutoEncoder app = new RunLoadExistAutoEncoder(
				"best/250_200_250.nt", "annt_parameters.json");
		SparkConf conf = CommonUtils.readSparkConf("annt_spark_conf.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		JavaRDD<UnLabeledDoubleSample> dataset = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/annt_train/trainning_norm/h26v05/b1");
		dataset = dataset.sample(true, 0.0006);
		dataset = dataset.cache();
		app.run(jsc, dataset, null);
		jsc.stop();
	}

	static {
		// 设置日志等级
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
		Logger.getLogger("apache").setLevel(Level.OFF);
	}
}
