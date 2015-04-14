package com.annt.app;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.jblas.DoubleMatrix;

import com.annt.obj.LabeledResult;
import com.annt.obj.UnLabeledDoubleSample;
import com.annt.utils.CommonUtils;

public class SetLabelApp {

	public static void main(String[] args) {
		SparkConf conf = CommonUtils.readSparkConf("default_spark.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		DoubleMatrix centers = CommonUtils.ReadDataset("center4.mt");
		final Broadcast<DoubleMatrix> bcenters = jsc.broadcast(centers);
		JavaRDD<UnLabeledDoubleSample> features = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/annt_train/features/h27v06/b1");
		features.map(new Function<UnLabeledDoubleSample, LabeledResult>() {

			private static final long serialVersionUID = 4849421412214193201L;

			public DoubleMatrix cs = bcenters.getValue();

			public LabeledResult call(UnLabeledDoubleSample v) throws Exception {
				int minIdx = -1;
				double min = Double.MAX_VALUE;
				DoubleMatrix sample = v.data;
				for (int i = 0; i < cs.columns; i++) {
					DoubleMatrix c = cs.getColumn(i);
					double distance = c.distance2(sample);
					if (distance < min) {
						min = distance;
						minIdx = i;
					}
				}
				return new LabeledResult(v.info, String.valueOf(minIdx));
			}
		})
				.repartition(13)
				.saveAsObjectFile(
						"hdfs://192.168.1.140:9000/user/terry/annt_train/result/h27v06/c4");
		jsc.stop();
	}

}
