package com.annt.junit;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import com.annt.obj.UnLabeledDoubleSample;
import com.annt.utils.CommonUtils;
import com.smiims.obj.GeoTSShortVector;

public class SparkANNTest {

	// @Test
	public void prepareDataset() {
		SparkConf conf = CommonUtils.readSparkConf("rbm_spark_conf.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		JavaRDD<GeoTSShortVector> vectorsRDD = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/ts_data/MODIS/2000049_2010353/MOD13/h23v04/b1");
		// 数据采样
		vectorsRDD = vectorsRDD.sample(true, 0.01);
		vectorsRDD
				.repartition(1)
				.saveAsObjectFile(
						"hdfs://192.168.1.140:9000/user/terry/ts_data/annt_train/trainning_orgin");
		jsc.stop();
	}

	@Test
	public void normDataset() {
		SparkConf conf = CommonUtils.readSparkConf("rbm_spark_conf.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		JavaRDD<GeoTSShortVector> vectorsRDD = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/ts_data/annt_train/trainning_orgin");
		System.out.println(vectorsRDD.count());
		JavaRDD<UnLabeledDoubleSample> result = vectorsRDD.map(new Function<GeoTSShortVector, UnLabeledDoubleSample>() {

			private static final long serialVersionUID = 1919807191769766098L;

			public UnLabeledDoubleSample call(GeoTSShortVector v)
					throws Exception {
				double data[] = new double[v.data.length];
				for (int i = 0; i < v.data.length; i++) {
					data[i] = (v.data[i] + 2000) / 12000;
				}
				UnLabeledDoubleSample result = new UnLabeledDoubleSample(
						new DoubleMatrix(data));
				return result;
			}
		});
		System.out.println(result.count());
		jsc.stop();
	}
}
