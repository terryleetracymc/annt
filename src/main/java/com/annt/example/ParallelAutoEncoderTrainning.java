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
				"best/17_13_17.nt", "annt_parameters.json");
		SparkConf conf = CommonUtils.readSparkConf("annt_spark_conf.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		JavaRDD<UnLabeledDoubleSample> dataset1 = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/annt_train/trainning/h27v06/select1l/b1");
		JavaRDD<UnLabeledDoubleSample> dataset2 = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/annt_train/trainning/h26v05/1year1l/b1");
		JavaRDD<UnLabeledDoubleSample> dataset = dataset1.union(dataset2);
		dataset.cache();
		for (int m = 0; m < 500; m++) {
			System.out.println(m);
			JavaRDD<UnLabeledDoubleSample> sub_dataset = dataset.sample(true,
					0.0005);
			sub_dataset = sub_dataset.repartition(app.app.groupNum).cache();
			app.run(jsc, sub_dataset, null);
		}
		jsc.stop();
	}

	static {
		// 设置日志等级
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
		Logger.getLogger("apache").setLevel(Level.OFF);
	}
}
