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
		SparkConf conf = CommonUtils.readSparkConf("default_spark.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		JavaRDD<UnLabeledDoubleSample> rdd = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/annt_train/trainning/h27v06/1year/b1");
		rdd = rdd.sample(true, 0.1);
		rdd = rdd.cache();
		List<UnLabeledDoubleSample> result = rdd.collect();
		int datasetSize = result.size();
		DoubleMatrix dataset = DoubleMatrix.zeros(23, datasetSize);
		for (int i = 0; i < result.size(); i++) {
			dataset.putColumn(i, result.get(i).data);
		}
		CommonUtils.SaveDataset("datasets/h27v06.mt", dataset);
		jsc.stop();
	}
}
