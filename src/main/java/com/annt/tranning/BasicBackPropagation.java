package com.annt.tranning;

import java.io.Serializable;
import java.util.LinkedList;

import org.jblas.DoubleMatrix;

import com.annt.interf.TrainingOperation;
import com.annt.layer.BasicLayer;
import com.annt.network.SimpleNetwork;

abstract public class BasicBackPropagation implements TrainingOperation,Serializable{

	/**
	 * 
	 */
	private static final long serialVersionUID = 5596999972018797481L;


	SimpleNetwork network;

	LinkedList<DoubleMatrix> weights;

	LinkedList<DoubleMatrix> biass;

	LinkedList<BasicLayer> layers;

	public LinkedList<DoubleMatrix> weights_updates;

	public LinkedList<DoubleMatrix> biass_updates;
	
	public void getUpdateMatrixs(DoubleMatrix input, DoubleMatrix ideal){
		updateMatrixAndBias(input, ideal);
	}
	
	abstract void updateMatrixAndBias(DoubleMatrix input, DoubleMatrix ideal);
}
