package com.annt.utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class CommonUtils {
	//读json文本文件全文
	@SuppressWarnings("resource")
	public String readJSONText(String path) throws FileNotFoundException{
		String content = new Scanner(new File(path)).useDelimiter("\\Z")
				.next();
		return content;
	}
}
