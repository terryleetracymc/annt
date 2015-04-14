package com.annt.app;

import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.jblas.DoubleMatrix;

import com.annt.obj.LabeledResult;
import com.annt.utils.CommonUtils;

public class SeeResult {

	public static void main(String[] args) {
		SparkConf conf = CommonUtils.readSparkConf("default_spark.json");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		JavaRDD<LabeledResult> results = jsc
				.objectFile("hdfs://192.168.1.140:9000/user/terry/annt_train/result/h27v06/c2");
		List<LabeledResult> imgStr = results.collect();
		DoubleMatrix dataset = new DoubleMatrix(4800, 4800);
		for (int i = 0; i < imgStr.size(); i++) {
			LabeledResult item = imgStr.get(i);
			String position = item.info;
			String label = item.label;
			int p = position.indexOf(",");
			int x = Integer.parseInt(position.substring(0, p));
			int y = Integer.parseInt(position.substring(p + 1));
			dataset.put(x, y, Double.parseDouble(label));
		}
		CommonUtils.SaveDataset("result2.mt", dataset);
		jsc.stop();
	}

}
