package com.annt.interf;

import org.apache.spark.api.java.JavaRDDLike;

public interface sparkOperation {
	@SuppressWarnings("rawtypes")
	JavaRDDLike operate(JavaRDDLike input,String args[]);
}
