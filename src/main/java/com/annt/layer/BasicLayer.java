package com.annt.layer;

import java.io.Serializable;

import com.annt.interf.ActiveFunction;

public class BasicLayer implements Serializable {

	/**
	 * 基础层
	 */
	private static final long serialVersionUID = 7712284174624153046L;
	public int neural_num;
	public boolean bias;
	public ActiveFunction activeFunc;

	// 定义本层的细胞数
	// 是否有偏置
	// 激活函数
	public BasicLayer(int n, boolean b, ActiveFunction f) {
		neural_num = n;
		bias = b;
		activeFunc = f;
	}
}
