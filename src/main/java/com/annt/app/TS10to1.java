package com.annt.app;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.jblas.DoubleMatrix;

import com.annt.obj.UnLabeledDoubleSample;
import com.annt.utils.CommonUtils;

public class TS10to1 {

	// 转10年时间序列为1年
	@SuppressWarnings("resource")
	public static void main(String[] args) {
		SparkConf conf = CommonUtils.readSparkConf("default_spark.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		JavaRDD<UnLabeledDoubleSample> dataset = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/annt_train/trainning/h27v06/10year/b1");
		dataset = dataset
				.map(new Function<UnLabeledDoubleSample, UnLabeledDoubleSample>() {
					private static final long serialVersionUID = -4974272316411125967L;

					public UnLabeledDoubleSample call(UnLabeledDoubleSample v)
							throws Exception {
						double d[] = new double[23];
						int offset = 0;
						for (int i = 0; i < d.length; i++) {
							d[i] = v.data.get(i + 20 + offset);
						}
						// d[0] = v.data.get(43);
						// d[5] = v.data.get(48);
						// d[7] = v.data.get(73);
						// d[9] = v.data.get(52);
						// d[12] = v.data.get(78);
						// d[14] = v.data.get(57);
						// d[15] = v.data.get(58);
						// d[16] = v.data.get(54);
						// d[17] = v.data.get(60);
						// d[18] = v.data.get(84);
						// d[20] = v.data.get(86);
						// d[21] = v.data.get(64);
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
						"hdfs://192.168.1.140:9000/user/terry/annt_train/trainning/h27v06/1year/b1");
		jsc.stop();
	}

}
