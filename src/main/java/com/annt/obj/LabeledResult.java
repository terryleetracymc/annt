package com.annt.obj;

import java.io.Serializable;

public class LabeledResult implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -819047440609357888L;

	public String info;
	public String label;

	public LabeledResult(String i, String l) {
		info = i;
		label = l;
	}
}
