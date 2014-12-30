package com.annt.tranning;

import java.util.LinkedList;

import org.jblas.DoubleMatrix;

import com.annt.interf.ActiveFunction;
import com.annt.layer.BasicLayer;
import com.annt.network.SimpleNetwork;

/**
 * 
 * @author terry 最简单的神经网络反向传播训练
 */

public class SimpleBackPropagation extends BasicBackPropagation {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3478496251202109693L;

	public SimpleBackPropagation(SimpleNetwork n) {
		network = n;
		weights = n.weights;
		biass = n.biass;
		layers = n.layers;
	}

	@Override
	// 训练一次
	public void updateMatrixAndBias(DoubleMatrix input, DoubleMatrix ideal) {
		// 获得每一层的输出
		LinkedList<DoubleMatrix> outputs = network.getOutputs(input);
		// 权值矩阵和偏置的更新
		weights_updates = new LinkedList<DoubleMatrix>();
		biass_updates = new LinkedList<DoubleMatrix>();
		int output_number = outputs.size();
		// 获得顶层信息
		// 首先计算顶层残差
		// 实际输出
		DoubleMatrix current_output = outputs.get(output_number - 1);
		BasicLayer current_layer = layers.get(output_number - 1);
		ActiveFunction current_function = current_layer.activeFunc;
		// 计算顶层残差
		DoubleMatrix current_error = current_output.sub(ideal).mul(
				current_function.derivative(current_output));
		// 计算更新的第一个矩阵
		// 获得下一层的输出
		current_output = outputs.get(output_number - 2);
		DoubleMatrix current_update = current_output.mmul(current_error
				.transpose());
		// 后插
		weights_updates.push(current_update);
		// System.out.println(current_error.rows + "*" + current_error.columns);
		// 用于计算残差的权值矩阵
		DoubleMatrix w = null;
		// 添加偏置更新
		if (current_layer.bias) {
			biass_updates.push(current_error);
		} else {
			biass_updates.push(null);
		}
		// 自顶向下更新
		for (int i = output_number - 2; i >= 1; i--) {
			current_layer = layers.get(i);
			// 获得该层的输出用于计算残差
			current_output = outputs.get(i);
			current_function = current_layer.activeFunc;
			w = weights.get(i);
			current_error = w.mmul(current_error).mul(
					current_function.derivative(current_output));
			// 获得下一层的输出用于计算更新矩阵
			current_output = outputs.get(i - 1);
			current_update = current_output.mmul(current_error.transpose());
			// 加入偏置
			if (current_layer.bias) {
				biass_updates.push(current_error);
			} else {
				biass_updates.push(null);
			}
			weights_updates.push(current_update);
		}
	}
}
