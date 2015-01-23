package com.annt.tranning;

import java.util.LinkedList;

import org.jblas.DoubleMatrix;

import com.annt.interf.ActiveFunction;
import com.annt.layer.BasicLayer;
import com.annt.network.SimpleNetwork;

public class SparseBackPropagation extends BasicBatBackPropagation {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5158530043820918359L;

	public LinkedList<DoubleMatrix> sparse_neural;

	public double expect_active = 0.05;

	public void setSparse(LinkedList<DoubleMatrix> sn) {
		sparse_neural = sn;
	}

	public void setExcepActive(double ea) {
		expect_active = ea;
	}

	// 构造函数
	public SparseBackPropagation(SimpleNetwork n) {
		super(n);
	}

	// 根据平均激活值和输入输出单次获得单次更新矩阵
	private void updateMatrixAndBiasWithSparse(DoubleMatrix input,
			DoubleMatrix ideal, LinkedList<DoubleMatrix> actives) {
		// 获得每一层的输出
		LinkedList<DoubleMatrix> outputs = network.getOutputs(input);
		// 权值和偏置更新
		weights_updates = new LinkedList<DoubleMatrix>();
		biass_updates = new LinkedList<DoubleMatrix>();
		int output_number = outputs.size();
		// 获得顶层信息
		// 首先计算顶层残差
		// 实际输出
		DoubleMatrix current_output = outputs.get(output_number - 1);
		BasicLayer current_layer = layers.get(output_number - 1);
		ActiveFunction current_function = current_layer.activeFunc;
		DoubleMatrix sn = sparse_neural.get(output_number - 1);
		DoubleMatrix active = actives.get(output_number - 1);
		DoubleMatrix sparse_error = sn.mul(active.rdiv(-expect_active).add(
				active.rsub(1).rdiv(1 - expect_active)));
		// 计算顶层残差
		DoubleMatrix current_error = (current_output.sub(ideal)
				.add(sparse_error)).mul(current_function
				.derivative(current_output));
		// 计算更新的第一个矩阵
		// 获得下一层的输出
		current_output = outputs.get(output_number - 2);
		DoubleMatrix current_update = current_output.mmul(current_error
				.transpose());
		// 后插
		weights_updates.push(current_update);
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
			sn = sparse_neural.get(i);
			active = actives.get(i);
			sparse_error = sn.mul(active.rdiv(-expect_active).add(
					active.rsub(1).rdiv(1 - expect_active)));
			w = weights.get(i);
			current_error = (w.mmul(current_error).add(sparse_error))
					.mul(current_function.derivative(current_output));
			// 获得下一层的输出用于计算更新矩阵
			current_output = outputs.get(i - 1);
			current_update = current_output.mmul(current_error.transpose());
			weights_updates.push(current_update);
			// 加入偏置
			if (current_layer.bias) {
				biass_updates.push(current_error);
			} else {
				biass_updates.push(null);
			}
		}
	}

	// 获得平均输出
	// 输入为多样本矩阵
	private LinkedList<DoubleMatrix> getAverageActive(DoubleMatrix inputs) {
		// 获得平均激活值
		DoubleMatrix input;
		BasicLayer l;
		LinkedList<DoubleMatrix> tmp_actives;
		LinkedList<DoubleMatrix> active_values = new LinkedList<DoubleMatrix>();
		// 激活值初始化
		for (int m = 0; m < layers.size(); m++) {
			l = layers.get(m);
			active_values.add(DoubleMatrix.zeros(l.neural_num));
		}
		// 激活值求和
		for (int m = 0; m < inputs.columns; m++) {
			input = inputs.getColumn(m);
			tmp_actives = network.getOutputs(input);
			for (int n = 0; n < active_values.size(); n++) {
				active_values.set(n,
						active_values.get(n).add(tmp_actives.get(n)));
			}
		}
		// 求平均激活值
		for (int m = 0; m < active_values.size(); m++) {
			active_values.set(m, active_values.get(m).div(inputs.columns));
		}
		return active_values;
	}

	// 输入为多样本矩阵
	@Override
	public void updateMatrixAndBias(DoubleMatrix inputs, DoubleMatrix ideals) {
		if (sparse_neural == null || sparse_neural.size() == 0) {
			System.err.println("warning:你没有设置稀疏参数，您可以直接使用"
					+ SimpleBatBackPropagation.class.getName() + "来训练神经网络");
			return;
		}
		// 变化权值矩阵和偏置变化向量的总和
		w_sum_update = new LinkedList<DoubleMatrix>();
		b_sum_update = new LinkedList<DoubleMatrix>();
		DoubleMatrix m = null, b = null;
		// 初始化
		for (int i = 0; i < weights.size(); i++) {
			m = weights.get(i);
			b = biass.get(i);
			w_sum_update.add(DoubleMatrix.zeros(m.rows, m.columns));
			if (b != null) {
				b_sum_update.add(DoubleMatrix.zeros(b.rows, b.columns));
			} else {
				b_sum_update.add(null);
			}
		}
		LinkedList<DoubleMatrix> actives = getAverageActive(inputs);
		// 根据平均激活值后向传播算法
		DoubleMatrix input;
		DoubleMatrix ideal;
		for (int i = 0; i < inputs.columns; i++) {
			// 获得样本
			input = inputs.getColumn(i);
			ideal = ideals.getColumn(i);
			updateMatrixAndBiasWithSparse(input, ideal, actives);
			// 权值变化叠加
			for (int j = 0; j < weights.size(); j++) {
				w_sum_update.set(j,
						w_sum_update.get(j).add(weights_updates.get(j)));
				if (b_sum_update.get(j) != null) {
					b_sum_update.set(j,
							b_sum_update.get(j).add(biass_updates.get(j)));
				}
			}
		}
		// 平均
		for (int j = 0; j < weights.size(); j++) {
			weights_updates.set(j, w_sum_update.get(j).div(inputs.columns));
			if (b_sum_update.get(j) != null) {
				biass_updates.set(j, b_sum_update.get(j).div(inputs.columns));
			}
		}
	}
}
