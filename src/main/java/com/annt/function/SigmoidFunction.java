package com.annt.function;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.annt.interf.ActiveFunction;

/**
 * @author terry Sigmoid
 */
public class SigmoidFunction implements ActiveFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2997571133636391147L;

	// 激活的激活值返回
	public DoubleMatrix active(DoubleMatrix w, DoubleMatrix x, DoubleMatrix b) {
		// 分有偏置和没有偏置的情况
		if (b != null) {
			return (w.transpose().mmul(x)).add(b);
		} else {
			return (w.transpose().mmul(x));
		}
	}

	// 输出值
	// sigmoid(x)=1/(1+exp(-active));
	public DoubleMatrix output(DoubleMatrix w, DoubleMatrix x, DoubleMatrix b) {
		DoubleMatrix active = active(w, x, b);
		DoubleMatrix output = MatrixFunctions.exp(active.rsub(0)).add(1)
				.rdiv(1);
		return output;
	}

	// 给定激活值输出
	public DoubleMatrix output(DoubleMatrix active) {
		return MatrixFunctions.exp(active.rsub(0)).add(1).rdiv(1);
	}

	// sigmoid'(x)=sigmoid(x)*(1-sigmoid(x))
	public DoubleMatrix derivative(DoubleMatrix w, DoubleMatrix x, DoubleMatrix b) {
		DoubleMatrix output = output(w, x, b);
		return output.mul(output.rsub(1));
	}

	//
	public DoubleMatrix derivative(DoubleMatrix output) {
		return output.mul(output.rsub(1));
	}

}
