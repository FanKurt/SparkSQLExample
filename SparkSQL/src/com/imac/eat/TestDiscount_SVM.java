package com.imac.eat;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.json.JSONException;
import org.json.JSONObject;

import scala.Tuple2;

public class TestDiscount_SVM {

	private static final org.apache.log4j.Logger LOGGER = org.apache.log4j.Logger
			.getLogger(TestDiscount.class);

	private static final String MYSQL_DRIVER = "com.mysql.jdbc.Driver";
	private static final String MYSQL_USERNAME = "root";
	private static final String MYSQL_PWD = "mysql";
	private static final String MYSQL_CONNECTION_URL = "jdbc:mysql://localhost:3306/eat?user="
			+ MYSQL_USERNAME + "&password=" + MYSQL_PWD;

	private static final JavaSparkContext sc = new JavaSparkContext(
			new SparkConf().setAppName("SparkJdbcDs").setMaster("local[*]"));

	private static final SQLContext sqlContext = new SQLContext(sc);

	private static ArrayList<String> discount_true = new ArrayList<>();
	private static ArrayList<String> discount_false = new ArrayList<>();

	public static void main(String[] args) throws JSONException {
		// Data source options
		Map<String, String> options = new HashMap<>();
		options.put("driver", MYSQL_DRIVER);
		options.put("url", MYSQL_CONNECTION_URL);
		options.put(
				"dbtable",
				"(select create_user,create_time from discount where locate('2015',create_time) > 0) as employees_name");
		options.put("partitionColumn", "create_user");
		options.put("lowerBound", "0");
		options.put("upperBound", "10");
		options.put("numPartitions", "10");

		// Load MySQL query result as DataFrame
		DataFrame jdbcDF = sqlContext.read().format("jdbc").options(options)
				.load();

		JavaRDD<String> rowRDD = jdbcDF.javaRDD()
				.map(new Function<Row, String>() {
					public String call(Row arg0) throws Exception {
						return arg0.get(0).toString();
					}
				}).distinct();

		List<String> user_id = rowRDD.collect();

		Map<String, String> options1 = new HashMap<>();
		options1.put("driver", MYSQL_DRIVER);
		options1.put("url", MYSQL_CONNECTION_URL);
		options1.put(
				"dbtable",
				"(select create_ip,account,create_time from user where locate('2015',create_time) > 0) as employees_name");
		options1.put("partitionColumn", "create_ip");
		options1.put("lowerBound", "0");
		options1.put("upperBound", "10");
		options1.put("numPartitions", "10");

		DataFrame jdbcDF1 = sqlContext.read().format("jdbc").options(options1)
				.load();

		JavaRDD<Row> rowRDD1 = jdbcDF1.javaRDD();

		List<Row> user_info = rowRDD1.collect();

		for (int i = 0; i < user_info.size(); i++) {
			if (user_info.get(i).get(1) != null) {
				try {
					String json_data = user_info.get(i).get(1).toString();
					JSONObject object = new JSONObject(json_data);
					String email = object.get("email").toString();
					String link = object.get("link").toString();
					String gender = object.get("gender").toString();
					String name = object.get("name").toString();
					String result_str = user_info.get(i).get(0) + "::" + email
							+ "::" + link + "::" + gender + "::" + name;
					if (isContainUserID(i, user_id)) {
						discount_true.add(i + "," + result_str + "," + 1.0);
					} else {
						discount_false.add(i + "," + result_str + "," + 0.0);
					}
				} catch (Exception e) {
				}
			}
		}

		JavaRDD<String> trueRDD = sc.parallelize(discount_true);
		JavaRDD<String> falseRDD = sc.parallelize(discount_false);
		JavaRDD<String> trainRDD = trueRDD.union(falseRDD);
		JavaRDD<String>[]  rdd = trainRDD.randomSplit(new double[]{0.8,0.2});
		JavaRDD<String> train_rdd = rdd[0];
		JavaRDD<String> test_rdd = rdd[1];
		
		train_rdd.cache();
		test_rdd.cache();
		
		JavaRDD<Tuple2<String, List<String>>> featrueRDD = train_rdd.map(new Function<String, Tuple2<String, List<String>>>() {
			public Tuple2<String, List<String>> call(String arg0)
					throws Exception {
				String [] token = arg0.split(",");
				String catkey = token[0]+"::"+token[token.length-1];
				List<String> catfeatures = Arrays.asList(token[1].split("::"));
				return new Tuple2<String, List<String>>(catkey, catfeatures);
			}
		});
		
		JavaRDD<ArrayList<Tuple2<Integer, String>>> train_cat_rdd = featrueRDD.map(new Function<Tuple2<String,List<String>>,ArrayList<Tuple2<Integer, String>>>() {
			public  ArrayList<Tuple2<Integer, String>> call(Tuple2<String, List<String>> arg0) throws Exception {
				return parseCatFeatures(arg0._2);
			}
		});
		
		
		final Map<Tuple2<Integer, String>, Long> oheMap = train_cat_rdd.flatMap(new FlatMapFunction<ArrayList<Tuple2<Integer, String>>,Tuple2<Integer, String>>() {
			public Iterable<Tuple2<Integer, String>> call(
					ArrayList<Tuple2<Integer, String>> arg0) throws Exception {
				return arg0;
			}
		}).distinct().zipWithIndex().collectAsMap();
		
		
		
		
		JavaRDD<LabeledPoint> ohe_train_rdd = featrueRDD.map(new Function<Tuple2<String,List<String>>, LabeledPoint>() {
			public LabeledPoint call(Tuple2<String, List<String>> arg0)
					throws Exception {
				ArrayList<Tuple2<Integer, String>> cat_features_indexed = parseCatFeatures(arg0._2);
				ArrayList<Double> cat_feature_ohe = new ArrayList<>();
				for(Tuple2<Integer, String> v : cat_features_indexed){
					if(oheMap.containsKey(v)){
						double b = (double)oheMap.get(v);
						cat_feature_ohe.add(b);
					}else{
						cat_feature_ohe.add(0.0);
					}
				}
				
				Object[] aa = cat_feature_ohe.toArray();
				double [] dd = new double[aa.length];
				for(int i=0 ;i<aa.length ; i++){
					dd[i] = (double) aa[i];
				}
				
				
				return new LabeledPoint(Double.parseDouble(arg0._1.split("::")[1]), Vectors.dense(dd));
			}
		});
		
		
		
		LOGGER.warn("SVMModel Start .....");
	
		int numIterations = 10;
	    final SVMModel model = SVMWithSGD.train(ohe_train_rdd.rdd(), numIterations);
	    
	    model.clearThreshold();
	    
	    
	    JavaRDD<Tuple2<String, List<String>>> test_raw_data = test_rdd.map(new Function<String, Tuple2<String, List<String>>>() {
			public Tuple2<String, List<String>> call(String arg0)
					throws Exception {
				String [] token = arg0.split(",");
				String catkey = token[0]+"::"+token[token.length-1];
				List<String> catfeatures = Arrays.asList(token[1].split("::"));
				return new Tuple2<String, List<String>>(catkey, catfeatures);
			}
		});
	    
	    JavaRDD<LabeledPoint> ohe_test_rdd = test_raw_data.map(new Function<Tuple2<String,List<String>>, LabeledPoint>() {
			public LabeledPoint call(Tuple2<String, List<String>> arg0)
					throws Exception {
				ArrayList<Tuple2<Integer, String>> cat_features_indexed = parseCatFeatures(arg0._2);
				ArrayList<Double> cat_feature_ohe = new ArrayList<>();
				for(Tuple2<Integer, String> v : cat_features_indexed){
					if(oheMap.containsKey(v)){
						double b = (double)oheMap.get(v);
						cat_feature_ohe.add(b);
					}else{
						cat_feature_ohe.add(0.0);
					}
				}
				
				Object[] aa = cat_feature_ohe.toArray();
				double [] dd = new double[aa.length];
				for(int i=0 ;i<aa.length ; i++){
					dd[i] = (double) aa[i];
				}
				
				
				return new LabeledPoint(Double.parseDouble(arg0._1.split("::")[1]), Vectors.dense(dd));
			}
		});
	    
		System.out.println(ohe_test_rdd.take(1));
//	    
//	    
//
//	    JavaRDD<Tuple2<Object, Object>> scoreAndLabels = ohe_test_rdd.map(
//	      new Function<LabeledPoint, Tuple2<Object, Object>>() {
//	        public Tuple2<Object, Object> call(LabeledPoint p) {
//	          Double score = model.predict(p.features());
//	          return new Tuple2<Object, Object>(score, p.label());
//	        }
//	      }
//	    );
//	    
//	    Double test_error = 1.0*scoreAndLabels.filter(new Function<Tuple2<Object,Object>, Boolean>() {
//			public Boolean call(Tuple2<Object, Object> arg0) throws Exception {
//				return !arg0._1().equals(arg0._2());
//			}
//		}).count()/ohe_test_rdd.count();
//	    
//	    System.out.println("============== Test Error =============");
//	    System.out.println(test_error);
	}
	public static Boolean isContainUserID(int i, List<String> user_id) {
		for (String r : user_id) {
			if (i == Integer.parseInt(r)) {
				return true;
			}
		}
		return false;
	}
	
	public static ArrayList<Tuple2<Integer, String>> parseCatFeatures(List<String> list){
		ArrayList<Tuple2<Integer, String>> arrayList = new ArrayList<>();
		for(int i=0 ; i <list.size();i++){
			arrayList.add(new Tuple2<Integer, String>(i, list.get(i)));
		}
		
		return arrayList;
	}
}
