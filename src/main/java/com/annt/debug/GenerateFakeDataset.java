package com.annt.debug;

import java.util.LinkedList;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.jblas.DoubleMatrix;

import com.annt.obj.DoubleSample;

/**
 * @author terry 生成假数据测试神经网络正确性
 */
public class GenerateFakeDataset {

	static String SPARK_URL = "spark://192.168.1.140:7077";
	static String HDFS_URL = "hdfs://192.168.1.140:9000/user/terry/";

	@SuppressWarnings("resource")
	public static void main(String[] args) {
		LinkedList<DoubleSample> dataset = new LinkedList<DoubleSample>();
		for (int i = 0; i < 50000; i++) {
			DoubleMatrix input = DoubleMatrix.randn(10);
			DoubleMatrix output = new DoubleMatrix(1);
			double result = input.mean();
			if (result > 1.0) {
				result = 1.0;
			} else if (result < 0.0) {
				result = 0.0;
			}
			output.put(0, result);
			dataset.add(new DoubleSample(input, output));
		}
		SparkConf conf = new SparkConf();
		conf.setMaster("local[4]");
		conf.setAppName("GenerateFakeDataset");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		jsc.parallelize(dataset).repartition(1)
				.saveAsObjectFile(HDFS_URL + "fakeDataset");
		;
		jsc.stop();
	}

}
