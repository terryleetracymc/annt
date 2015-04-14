package com.annt.app;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.jblas.DoubleMatrix;

import com.annt.network.SimpleNetwork;
import com.annt.obj.UnLabeledDoubleSample;
import com.annt.utils.CommonUtils;

public class GenerateFeaturesApp extends SparkApp {

	/**
	 * 使用神经网络生成特征值
	 */
	private static final long serialVersionUID = -4926119575925516932L;

	public GenerateFeaturesApp() {
	}

	// 获得特征
	public JavaRDD<UnLabeledDoubleSample> getFeatures(
			final Broadcast<SimpleNetwork> bNetwork,
			JavaRDD<UnLabeledDoubleSample> dataset) {
		return dataset.map(
				new Function<UnLabeledDoubleSample, UnLabeledDoubleSample>() {

					/**
			 * 
			 */
					private static final long serialVersionUID = 8433612114003961304L;

					SimpleNetwork network = bNetwork.getValue();

					public UnLabeledDoubleSample call(UnLabeledDoubleSample v)
							throws Exception {
						DoubleMatrix sample = v.data;
						DoubleMatrix feature = network.getOutputs(sample)
								.get(1);
						UnLabeledDoubleSample output = new UnLabeledDoubleSample(
								feature);
						output.info = v.info;
						return output;
					}
				}).filter(new Function<UnLabeledDoubleSample, Boolean>() {
			/**
			 * 
			 */
			private static final long serialVersionUID = -1151226254190320127L;

			public Boolean call(UnLabeledDoubleSample v) throws Exception {
				if (v == null) {
					return false;
				}
				return true;
			}
		});
	}

	@SuppressWarnings("resource")
	public static void main(String args[]) {
		SparkConf conf = CommonUtils
				.readSparkConf("generate_features_conf.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		JavaRDD<UnLabeledDoubleSample> samples = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/annt_train/trainning/h26v05/1year/b1");
		final Broadcast<SimpleNetwork> bNetwork = jsc.broadcast(SimpleNetwork
				.loadNetwork("best/23_17_23.nt"));
		GenerateFeaturesApp app = new GenerateFeaturesApp();
		JavaRDD<UnLabeledDoubleSample> features = app.getFeatures(bNetwork,
				samples);
		// 78/62
		features.repartition(78)
				.saveAsObjectFile(
						"hdfs://192.168.1.140:9000/user/terry/annt_train/trainning/h26v05/1year1l/b1");
		jsc.stop();
	}
}
