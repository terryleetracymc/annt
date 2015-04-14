package com.annt.app;

import java.io.Serializable;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.jblas.DoubleMatrix;

import com.annt.obj.UnLabeledDoubleSample;
import com.annt.utils.CommonUtils;

class GrabInteger implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -5681379993800497089L;
	public int i;

	public GrabInteger(int idx) {
		i = idx;
	}

	public void setI(int idx) {
		i = idx;
	}
}

public class GenerateImg {

	@SuppressWarnings("resource")
	public static void main(String[] args) {
		SparkConf conf = CommonUtils.readSparkConf("default_spark.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		final Broadcast<GrabInteger> bs = jsc.broadcast(new GrabInteger(0));
		JavaRDD<UnLabeledDoubleSample> rdd = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/annt_train/restore/h27v06/1year/b1");
		rdd = rdd.cache();
		for (int i = 0; i < 23; i++) {
			bs.getValue().setI(i);
			JavaRDD<UnLabeledDoubleSample> n_rdd = rdd
					.map(new Function<UnLabeledDoubleSample, UnLabeledDoubleSample>() {

						private static final long serialVersionUID = -8772405713061711622L;

						int idx = bs.getValue().i;

						public UnLabeledDoubleSample call(
								UnLabeledDoubleSample v) throws Exception {
							double d[] = new double[1];
							d[0] = v.data.get(idx);
							DoubleMatrix data = new DoubleMatrix(d);
							UnLabeledDoubleSample sample = new UnLabeledDoubleSample(
									data);
							sample.info = v.info;
							return sample;
						}
					}).cache();
			DoubleMatrix dataset = new DoubleMatrix(4800, 4800);
			List<UnLabeledDoubleSample> imgStr = n_rdd.collect();
			n_rdd.unpersist();
			for (int j = 0; j < imgStr.size(); j++) {
				UnLabeledDoubleSample item = imgStr.get(j);
				String position = item.info;
				int p = position.indexOf(",");
				int x = Integer.parseInt(position.substring(0, p));
				int y = Integer.parseInt(position.substring(p + 1));
				dataset.put(x, y, item.data.get(0));
			}
			CommonUtils.SaveDataset("result/h27v06/" + i + ".mt", dataset);
		}
		jsc.stop();
	}

}
