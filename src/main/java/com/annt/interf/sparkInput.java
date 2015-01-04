package com.annt.interf;

import org.apache.spark.api.java.JavaRDDLike;
import org.apache.spark.api.java.JavaSparkContext;

public interface sparkInput {
	@SuppressWarnings("rawtypes")
	JavaRDDLike input(JavaSparkContext jsc,String args[]);
}
