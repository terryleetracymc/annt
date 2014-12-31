package com.annt.tranning;

import java.util.LinkedList;

import org.jblas.DoubleMatrix;

import com.annt.network.SimpleNetwork;

public class SimpleBatBackPropagation extends BasicBatBackPropagation {

	/**
	 * 反向传播批训练
	 */
	private static final long serialVersionUID = 4704221269423094177L;

	public SimpleBatBackPropagation(SimpleNetwork n) {
		network = n;
		weights = n.weights;
		biass = n.biass;
		layers = n.layers;
	}

	// 输入为多样本矩阵
	@Override
	public void updateMatrixAndBias(DoubleMatrix inputs, DoubleMatrix ideals) {
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
		// 一列为一个样本
		//
		DoubleMatrix input;
		DoubleMatrix ideal;
		for (int i = 0; i < inputs.columns; i++) {
			input = inputs.getColumn(i);
			ideal = ideals.getColumn(i);
			// 调用一次训练函数函数
			super.updateMatrixAndBias(input, ideal);
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
