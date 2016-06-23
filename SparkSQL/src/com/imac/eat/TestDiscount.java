package com.imac.eat;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import org.apache.spark.Partition;
import org.apache.spark.SparkConf;
import org.apache.spark.TaskContext;
import org.apache.spark.annotation.DeveloperApi;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.DoubleFlatMapFunction;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function3;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.examples.ml.Document;
import org.apache.spark.examples.ml.LabeledDocument;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.RFormula;
import org.apache.spark.ml.feature.RFormulaModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.feature.StandardScaler;
import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset;
import org.apache.spark.mllib.fpm.FPGrowthModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.GradientBoostedTrees;
import org.apache.spark.mllib.tree.configuration.BoostingStrategy;
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import com.google.common.collect.Lists;

import scala.Tuple2;
import scala.Tuple3;
import scala.collection.Iterator;
import scala.collection.mutable.ListBuffer;

/**
 * 
 * @author user88
 * 
 */
public class TestDiscount implements Serializable {

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
		
		
		System.out.println("GradientBoostedTreesModel Start .....");
	    BoostingStrategy boostingStrategy = BoostingStrategy.defaultParams("Classification");
	    boostingStrategy.setNumIterations(100); // Note: Use more iterations in practice.
	    boostingStrategy.getTreeStrategy().setNumClasses(2);
	    boostingStrategy.getTreeStrategy().setMaxDepth(5);
	    

	    final GradientBoostedTreesModel model =GradientBoostedTrees.train(ohe_train_rdd, boostingStrategy);
	  
	    
	    //test....
	    
	    JavaRDD<Tuple2<String, List<String>>> test_raw_data = test_rdd.map(new Function<String, Tuple2<String, List<String>>>() {
			public Tuple2<String, List<String>> call(String arg0)
					throws Exception {
				String [] token = arg0.split(",");
				String catkey = token[0]+"::"+token[token.length-1];
				List<String> catfeatures = Arrays.asList(token[1].split("::"));
				return new Tuple2<String, List<String>>(catkey, catfeatures);
			}
		});
	    
	    JavaRDD<LabeledPoint> testData  = test_raw_data.map(new Function<Tuple2<String,List<String>>, LabeledPoint>() {
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
	    
	    
	    JavaPairRDD<Double, Double> predictionAndLabel =
	      testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
	        public Tuple2<Double, Double> call(LabeledPoint p) {
	          return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
	        }
	      });
	    
	    Double testErr =1.0 * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
	        public Boolean call(Tuple2<Double, Double> pl) {
	          return !pl._1().equals(pl._2());
	        }
	      }).count() / testData.count();
	    
	    System.out.println("Test Error: " + testErr);
	    System.out.println("Learned classification GBT model:\n" + model.toDebugString());
		
//		JavaRDD<LabeledDocument> labelRDD = trainRDD.map(new Function<String, LabeledDocument>() {
//			public LabeledDocument call(String arg0) throws Exception {
//				String [] token = arg0.split(",");
//				return new LabeledDocument(Long.parseLong(token[0]), token[1], Double.parseDouble(token[2]));
//			}
//		});
//
//		DataFrame training = sqlContext.createDataFrame(labelRDD, LabeledDocument.class);
//
//		Tokenizer tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words");
//		
//		HashingTF hashingTF = new HashingTF().setNumFeatures(1000)
//				.setInputCol(tokenizer.getOutputCol()).setOutputCol("features");
//		
//		LogisticRegression lr = new LogisticRegression().setMaxIter(10)
//				.setRegParam(0.001);
//		
//		Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {
//				tokenizer, hashingTF, lr });
//
//		PipelineModel model = pipeline.fit(training);
//		
//		
//		List<Document> localTest = Lists.newArrayList(
//			      new Document(4L, "1.168.80.164 k753357@yahoo.com.tw https://www.facebook.com/profile.php?id=100000193098738 male ­S´Óµ¾"),
//			      new Document(5L, "10.21.20.235 a0936196693@gmail.com https://www.facebook.com/profile.php?id=100001903790527&fref=ts male §dªÃ¿Ù"),
//			      new Document(6L, "163.17.131.225 p891411@yahoo.com.tw https://www.facebook.com/app_scoped_user_id/626232574183902/ female Ting Meng Hsu"),
//			      new Document(7L, "101.15.210.239 s1499k005@gmail.com https://www.facebook.com/kairen.bai.1?fref=ts male ¥Õ³Í¤¯"));
//			    
//		DataFrame test = sqlContext.createDataFrame(sc.parallelize(localTest), Document.class);
//		
//	    DataFrame predictions = model.transform(test);
//	    for (Row r: predictions.select("id", "text", "probability", "prediction").collect()) {
//	      System.out.println("(" + r.get(0) + ", " + r.get(1) + ") -->  prediction=" + r.get(3));
//	    }

		sc.stop();
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
