package com.annt.app;

import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.jblas.DoubleMatrix;

import com.annt.obj.UnLabeledDoubleSample;
import com.annt.utils.CommonUtils;

public class GetLocalSample {
	@SuppressWarnings("resource")
	public static void main(String args[]) {
		SparkConf conf = CommonUtils.readSparkConf("annt_spark_conf.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		JavaRDD<UnLabeledDoubleSample> rdd = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/annt_train/trainning_norm/h26v05/b1");
		rdd = rdd.sample(true, 0.0001);
		rdd = rdd.cache();
		long datasetSize = rdd.count();
		DoubleMatrix dataset = DoubleMatrix.zeros(250, (int) datasetSize);
		List<UnLabeledDoubleSample> result = rdd.collect();
		for (int i = 0; i < result.size(); i++) {
			dataset.putColumn(i, result.get(i).data);
		}
		CommonUtils.SaveDataset("test_sample.mt", dataset);
		jsc.stop();
	}
}
