package com.annt.app;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.jblas.DoubleMatrix;

import com.annt.obj.UnLabeledDoubleSample;
import com.annt.utils.CommonUtils;

public class SparkKmean {

	static {
		// 设置日志等级
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
		Logger.getLogger("apache").setLevel(Level.OFF);
	}

	public static void main(String[] args) {
		SparkConf conf = CommonUtils.readSparkConf("default_spark.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);

		JavaRDD<UnLabeledDoubleSample> dataset = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/annt_train/features/h27v06/b1");
		JavaRDD<Vector> vdataset = dataset
				.map(new Function<UnLabeledDoubleSample, Vector>() {
					private static final long serialVersionUID = -7218053393886577677L;

					public Vector call(UnLabeledDoubleSample v)
							throws Exception {
						return Vectors.dense(v.data.data);
					}
				});
		vdataset = vdataset.cache();
		int numClusters = 10;
		int numIterations = 200;
		KMeansModel clusters = KMeans.train(vdataset.rdd(), numClusters,
				numIterations);
		Vector[] centers = clusters.clusterCenters();
		DoubleMatrix cenvs = new DoubleMatrix(10, numClusters);
		for (int i = 0; i < centers.length; i++) {
			cenvs.putColumn(i, new DoubleMatrix(centers[i].toArray()));
		}
		CommonUtils.SaveDataset("center4.mt", cenvs);
		jsc.stop();
	}
}
