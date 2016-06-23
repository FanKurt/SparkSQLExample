package com.imac;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.hive.HiveContext;

public class JavaSparkSQLHive {
	public static void main(String[] args) {
		JavaSparkContext sc = new JavaSparkContext();
		HiveContext sqlContext = new org.apache.spark.sql.hive.HiveContext(sc.sc());
		String path = args[0];
		String table = args[1];
		sqlContext.sql("CREATE TABLE IF NOT EXISTS "+table+" (key INT, value STRING)");
		sqlContext.sql("LOAD DATA INPATH '"+path+"' INTO TABLE "+table);

		// Queries are expressed in HiveQL.
		Row[] results = sqlContext.sql("SELECT * FROM "+table).collect();
		
		
		for(Row v : results){
			System.out.println(v);
		}
	}

}
