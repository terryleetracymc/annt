package com.annt.app;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.jblas.DoubleMatrix;

import com.annt.obj.UnLabeledDoubleSample;
import com.annt.utils.CommonUtils;

public class SelectSampleApp {

	@SuppressWarnings("resource")
	public static void main(String[] args) {
		SparkConf conf = CommonUtils.readSparkConf("default_spark.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		JavaRDD<UnLabeledDoubleSample> dataset = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/annt_train/trainning/h27v06/10year/b1");
		dataset = dataset
				.map(new Function<UnLabeledDoubleSample, UnLabeledDoubleSample>() {

					private static final long serialVersionUID = 1529159095355008133L;

					public UnLabeledDoubleSample call(UnLabeledDoubleSample v)
							throws Exception {
						// 选取10年周期中最大的数作为样本输入
						double d[] = new double[23];
						for (int i = 0; i < d.length; i++) {
							double max_value = Double.MIN_VALUE;
							// 两年选最大
							for (int j = i; j < 23 * 4; j = j + 23) {
								if (v.data.get(j + 20) > max_value) {
									max_value = v.data.get(j + 20);
								}
							}
							d[i] = max_value;
						}
						DoubleMatrix data = new DoubleMatrix(d);
						UnLabeledDoubleSample sample = new UnLabeledDoubleSample(
								data);
						sample.info = v.info;
						return sample;
					}
				});
		dataset.repartition(90)
				.cache()
				.saveAsObjectFile(
						"hdfs://192.168.1.140:9000/user/terry/annt_train/trainning/h27v06/select/b1");
		jsc.stop();
	}

}
