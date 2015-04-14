package com.annt.app;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.jblas.DoubleMatrix;

import com.annt.obj.UnLabeledDoubleSample;
import com.annt.utils.CommonUtils;
import com.smiims.obj.GeoTSShortVector;

public class GenerateNorm {

	@SuppressWarnings("resource")
	public static void main(String[] args) {
		SparkConf conf = CommonUtils.readSparkConf("rbm_spark_conf.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		JavaRDD<GeoTSShortVector> vectorsRDD = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/ts_data/MODIS/2000049_2010353/MOD13/h26v05/b1");
		// System.out.println(vectorsRDD.count());
		JavaRDD<UnLabeledDoubleSample> result = vectorsRDD
				.map(new Function<GeoTSShortVector, UnLabeledDoubleSample>() {

					private static final long serialVersionUID = 1919807191769766098L;

					public UnLabeledDoubleSample call(GeoTSShortVector v)
							throws Exception {
						double data[] = new double[v.data.length];
						for (int i = 0; i < v.data.length; i++) {
							data[i] = v.data[i];
						}
						UnLabeledDoubleSample result = new UnLabeledDoubleSample(
								new DoubleMatrix(data));
						result.info = v.x + "," + v.y;
						return result;
					}
				});
		result = result
				.map(new Function<UnLabeledDoubleSample, UnLabeledDoubleSample>() {

					/**
					 * 
					 */
					private static final long serialVersionUID = -6097747044994206297L;

					public UnLabeledDoubleSample call(UnLabeledDoubleSample v)
							throws Exception {
						v.data.addi(2000).divi(12000);
						return v;
					}
				});
		result.repartition(750)
				.saveAsObjectFile(
						"hdfs://192.168.1.140:9000/user/terry/annt_train/trainning/h26v05/10year/b1");
		jsc.stop();
	}

}