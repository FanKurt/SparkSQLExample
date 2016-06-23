package com.imac.eat;


import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset;
import org.apache.spark.mllib.fpm.FPGrowthModel;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.json.JSONArray;
import org.json.JSONObject;

import scala.Tuple2;

/**
 * 進入 Recommond 前，使用者操作畫面順序之關聯式分析 
 * @author user88
 *
 */
public class FPGrowth_Feet implements Serializable {

    private static final org.apache.log4j.Logger LOGGER = org.apache.log4j.Logger.getLogger(FPGrowth_Feet.class);

    private static final String MYSQL_DRIVER = "com.mysql.jdbc.Driver";
    private static final String MYSQL_USERNAME = "root";
    private static final String MYSQL_PWD = "mysql";
    private static final String MYSQL_CONNECTION_URL =
            "jdbc:mysql://localhost:3306/eat?user=" + MYSQL_USERNAME + "&password=" + MYSQL_PWD;

    private static final JavaSparkContext sc =
            new JavaSparkContext(new SparkConf().setAppName("SparkJdbcDs").setMaster("local[*]"));

    private static final SQLContext sqlContext = new SQLContext(sc);

    public static void main(String[] args) {
        //Data source options
        Map<String, String> options = new HashMap<>();
        options.put("driver", MYSQL_DRIVER);
        options.put("url", MYSQL_CONNECTION_URL);
        options.put("dbtable","(select uid,feet,create_time from feet where locate('2015',create_time) > 0) as employees_name");
        options.put("partitionColumn", "uid");
        options.put("lowerBound", "0");
        options.put("upperBound", "10");
        options.put("numPartitions", "10");
        
        //Load MySQL query result as DataFrame
        DataFrame jdbcDF = sqlContext.read().format("jdbc").options(options).load();
        
        JavaRDD<Row> rowRDD = jdbcDF.javaRDD();
        JavaRDD<List<String>> transactions =  rowRDD.filter(new Function<Row, Boolean>() {
			public Boolean call(Row arg0) throws Exception {
				return arg0.get(1).toString().contains("recommend");
			}
		}).map(new Function<Row, String>() {
			public String call(Row arg0) throws Exception {
				return arg0.get(1).toString();
			}
		}).map(new Function<String, String>() {
			public String call(String arg0) throws Exception {
				ArrayList<Tuple2<String, Integer>> arList = new ArrayList<Tuple2<String, Integer>>();
				String str = "";
				try{
					JSONArray array = new JSONArray(arg0);
					for(int i=1 ; i< array.length() ;i++){
						JSONObject object = new JSONObject(array.get(i).toString());
						String time = object.get("t").toString();
						String page = object.getJSONObject("feet").get("page").toString();
						String type = object.getJSONObject("feet").get("type").toString();
						if(type.equals("pageshow")){
							if(!page.equals("recommend")){
								if(!str.contains(page)){
									str+=page+",";
								}
							}else{
								return str.substring(0, str.length()-1);
							}
						}
					}
				}catch(Exception e){
				}
				return "";
			}
		}).filter(new Function<String, Boolean>() {
			public Boolean call(String arg0) throws Exception {
				return !arg0.equals("");
			}
		}).map(new Function<String, List<String>>() {
			public List<String> call(String arg0) throws Exception {
				ArrayList<String> arrList = new ArrayList<>();
				String [] token = arg0.split(",");
				for(String value : token){
					arrList.add(value);
				}
				return arrList;
			}
		});
        
        FPGrowth fpg = new FPGrowth().setMinSupport(0.2).setNumPartitions(10);
		FPGrowthModel<String> model = fpg.run(transactions);

		for(FreqItemset<String> v : model.freqItemsets().toJavaRDD().collect()){
			System.out.println(v.javaItems()+"			"+v.freq());
		}
        
    }
}
