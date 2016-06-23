package com.imac;

import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.serializer.KryoSerializer;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.elasticsearch.spark.rdd.api.java.JavaEsSpark;
import org.json.simple.JSONObject;

import scala.Tuple2;

public class SparkSQL_ES {
	public static void main(String[] args) {
		if (args.length < 1) {
			System.out.println("Format Error : [Receive ES index]");
		}
		SparkConf conf = new SparkConf().setMaster("local[2]").setAppName(
				"Elasticsearch");
		conf.set("spark.serializer", KryoSerializer.class.getName());
		conf.set("es.index.auto.create", "true");
		conf.set("es.nodes", "10.26.1.17:9200");
		conf.set("es.input.json", "true");

		JavaSparkContext sc = new JavaSparkContext(conf);
		SQLContext sqlContext = new SQLContext(sc);
		
		JavaPairRDD<String, Map<String, Object>> esRDD = JavaEsSpark.esRDD(sc,args[0]);
		JavaRDD<String> json_data = esRDD.map(new Function<Tuple2<String, Map<String, Object>>, String>() {
					public String call(Tuple2<String, Map<String, Object>> arg0)
							throws Exception {
						JSONObject json = new JSONObject();
						Map<String, Object> map = arg0._2;
						json.putAll(map);
						return json.toString();
					}
				});
		DataFrame json_frame = sqlContext.jsonRDD(json_data);
		json_frame.printSchema();

//		json_frame.registerTempTable("json");
//
//		DataFrame message = sqlContext.sql("SELECT message FROM json");
//		
//		List<String> nameAndCity = message.toJavaRDD().map(new Function<Row, String>() {
//					@Override
//					public String call(Row row) {
//						return "Name: " + row.getString(0);
//					}
//				}).collect();
		long init_time = System.currentTimeMillis();
		DataFrame message = json_frame.select("message");
//		Row [] row =message.collect();
//		for(Row r : row){
//			if(r.getString(0).contains("Scheduling")){
//				System.out.println("row  "+r.getString(0));
//			}
//		}
		long end_time = System.currentTimeMillis();
		System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
		System.out.println("time   "+(end_time-init_time)/1000 + " s");
		System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");

//		json_data.saveAsTextFile("/spark_sql_es_out");
		sc.stop();

	}

}
