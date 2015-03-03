package com.annt.utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Scanner;
import java.util.Set;

import org.apache.spark.SparkConf;
import org.jblas.DoubleMatrix;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.annt.evaluate._2LEvaluate;
import com.annt.network.RBMNetwork;
import com.annt.network.SimpleNetwork;
import com.annt.obj.NetworkUpdateParameters;
import com.annt.trainning.CDKBackPropagation;
import com.annt.trainning.SimpleBackPropagation;

public class CommonUtils {
	// 读json文本文件全文
	public static String readJSONText(String path) {
		String content = "{}";
		try {
			content = new Scanner(new File(path)).useDelimiter("\\Z").next();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return content;
	}

	// 读取数据集
	public static DoubleMatrix ReadDataset(String path) {
		try {
			ObjectInputStream in;
			in = new ObjectInputStream(new FileInputStream(path));
			DoubleMatrix result = (DoubleMatrix) in.readObject();
			in.close();
			return result;
		} catch (FileNotFoundException e) {
			// log
			e.printStackTrace();
		} catch (IOException e) {
			// log
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// log
			e.printStackTrace();
		}
		return null;
	}

	// 保存数据集
	public static void SaveDataset(String path, DoubleMatrix dataset) {
		try {
			ObjectOutputStream out;
			out = new ObjectOutputStream(new FileOutputStream(path));
			out.writeObject(dataset);
			out.close();
		} catch (FileNotFoundException e) {
			// log
			e.printStackTrace();
		} catch (IOException e) {
			// log
			e.printStackTrace();
		}
	}

	// 存储图像到指定路径
	public static void SaveImgToPath(String path, double data[], int width,
			int height) {
		Mat img = new Mat(width, height, CvType.CV_64FC1);
		for (int m = 0; m < width; m++) {
			for (int n = 0; n < height; n++) {
				img.put(m, n, data[m + width * n]);
			}
		}
		Highgui.imwrite(path, img);
	}

	// 自编码器
	// 根据样本使用BP算法训练自编码器神经网络
	public static void GetTargetNetwork(DoubleMatrix dataset,
			SimpleNetwork network, double max_error, int time,
			double learning_rate, double lamda, String bestPath) {
		DoubleMatrix sample = null;
		SimpleBackPropagation sbp = new SimpleBackPropagation(network);
		NetworkUpdateParameters ups = new NetworkUpdateParameters(network);
		double min_error = Double.MAX_VALUE;
		ups.zeroAll();
		for (int m = 0; m < time; m++) {
			for (int i = 0; i < dataset.columns; i++) {
				sample = dataset.getColumn(i);
				sbp.updateMatrixAndBias(sample, sample);
				ups.addAll(sbp.weights_updates, sbp.biass_updates);
			}
			ups.div(dataset.columns);
			ups.addLamdaWeights(lamda, network.weights);
			network.updateNet(ups.wus, ups.bus, learning_rate);
			// 计算误差
			double error = 0.0;
			_2LEvaluate evaluate = new _2LEvaluate();
			for (int i = 0; i < dataset.columns; i++) {
				sample = dataset.getColumn(i);
				error += evaluate.getError(sample, network.getOutput(sample));
			}
			error = error / dataset.columns;
			System.out.println((m + 1) + " : " + error);
			if (error < min_error) {
				SimpleNetwork.saveNetwork(bestPath, network);
				min_error = error;
			}
			if (error < max_error) {
				break;
			}
		}
		System.out.println(min_error);
	}

	// 根据样本使用CD-K算法初始化RBM
	public static void GetTargetRBM(DoubleMatrix dataset, RBMNetwork rbm,
			double max_error, int time, double learning_rate, int cd_k,
			String bestPath) {
		DoubleMatrix sample = null;
		CDKBackPropagation cdkBP = new CDKBackPropagation(rbm);
		cdkBP.setK(cd_k);
		double min_error = Double.MAX_VALUE;
		for (int m = 0; m < time; m++) {
			DoubleMatrix wu = DoubleMatrix.zeros(rbm.vn, rbm.hn);
			DoubleMatrix vu = DoubleMatrix.zeros(rbm.vn);
			DoubleMatrix hu = DoubleMatrix.zeros(rbm.hn);
			// 获得权值更新
			for (int i = 0; i < dataset.columns; i++) {
				sample = dataset.getColumn(i);
				cdkBP.updateMatrixAndBias(sample);
				wu.addi(cdkBP.wu);
				vu.addi(cdkBP.vbu);
				hu.addi(cdkBP.hbu);
			}
			// 平均权值更新
			wu.divi(dataset.columns);
			vu.divi(dataset.columns);
			hu.divi(dataset.columns);
			rbm.updateRBM(wu, vu, hu, learning_rate);
			// 计算误差
			double error = 0.0;
			_2LEvaluate evaluate = new _2LEvaluate();
			for (int i = 0; i < dataset.columns; i++) {
				sample = dataset.getColumn(i);
				DoubleMatrix restoreSign = rbm.getVOutput(rbm
						.getHOutput(sample));
				error += evaluate.getError(sample, restoreSign);
			}
			error = error / dataset.columns;
			System.out.println((m + 1) + " : " + error);
			if (error < min_error) {
				RBMNetwork.saveNetwork(bestPath, rbm);
				min_error = error;
			}
			if (error < max_error) {
				break;
			}
		}
		System.out.println(min_error);
	}

	// 读取Spark配置文件
	public static SparkConf readSparkConf(String path) {
		String jsonStr = readJSONText(path);
		JSONObject conf_json = JSONObject.parseObject(jsonStr);
		SparkConf conf = new SparkConf();
		conf.setAppName(conf_json.getString("SPARK_APPNAME"));
		conf.setMaster(conf_json.getString("SPARK_URL"));
		JSONObject spark_base_conf = conf_json.getJSONObject("SPARK_BASE_CONF");
		Set<String> keys = spark_base_conf.keySet();
		for (String key : keys) {
			conf.set(key, spark_base_conf.getString(key));
		}
		// 添加jar包
		JSONArray jars_array = conf_json.getJSONArray("jars");
		String[] jars_conf = new String[jars_array.size()];
		for (int i = 0; i < jars_conf.length; i++) {
			jars_conf[i] = jars_array.getString(i);
		}
		conf.setJars(jars_conf);
		return conf;
	}
}
