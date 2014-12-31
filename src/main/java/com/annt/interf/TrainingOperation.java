package com.annt.interf;

import org.jblas.DoubleMatrix;

/**
 * @author terry 神经网络训练的操作
 */
public interface TrainingOperation {

	void getUpdateMatrixs(DoubleMatrix input, DoubleMatrix ideal);
}
