package com.imac;

import java.io.Serializable;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.wltea.analyzer.core.IKSegmenter;
import org.wltea.analyzer.core.Lexeme;

import scala.Tuple2;


public class TranewsKeyWord implements Serializable {

    private static final org.apache.log4j.Logger LOGGER = org.apache.log4j.Logger.getLogger(TranewsKeyWord.class);

    private static final String MYSQL_DRIVER = "com.mysql.jdbc.Driver";
    private static final String MYSQL_USERNAME = "manager";
    private static final String MYSQL_PWD = "LtJevdtz5X6zU4fWsWTccOXn0UXv3IpT0PJ65M6l6yHj058apT";
    private static final String MYSQL_CONNECTION_URL =
            "jdbc:mysql://211.23.17.100:3306/itravel?user=" + MYSQL_USERNAME + "&password=" + MYSQL_PWD;

    private static final JavaSparkContext sc =
            new JavaSparkContext(new SparkConf().setAppName("SparkJdbcDs").setMaster("local[*]"));

    private static final SQLContext sqlContext = new SQLContext(sc);

    public static void main(String[] args) {
        //Data source options
        Map<String, String> options = new HashMap<>();
        options.put("driver", MYSQL_DRIVER);
        options.put("url", MYSQL_CONNECTION_URL);
        options.put("dbtable","(select time,title,content from news ) as employees_name");
        options.put("partitionColumn", "time");
        options.put("lowerBound", "0");
        options.put("upperBound", "10");
        options.put("numPartitions", "10");
        
        DataFrame jdbcDF = sqlContext.read().format("jdbc").options(options).load();
        
        JavaRDD<Row> documents = jdbcDF.select("content").toJavaRDD().filter(new Function<Row, Boolean>() {
			public Boolean call(Row arg0) throws Exception {
				return !arg0.get(0).toString().equals("");
			}
		}).map(new Function<Row, List<String>>() {
			public List<String> call(Row arg0) throws Exception {
				ArrayList<String> arrayList = new ArrayList<>();
				StringReader reader = new StringReader(arg0.get(0).toString().trim());

				IKSegmenter ik = new IKSegmenter(reader, true);
				Lexeme lexeme = null;
				while ((lexeme = ik.next()) != null) {
					if(lexeme.getLexemeText().length()>=2){
						arrayList.add(lexeme.getLexemeText());
					}
				}
				return arrayList;
			}
		}).map(new Function<List<String>, Row>() {
			public Row call(List<String> arg0) throws Exception {
				String str = "";
				for(String v : arg0){
					str+=v+" ";
				}
				return RowFactory.create(str.substring(0, str.length()-1));
			}
		});
        
        SQLContext sqlContext = new SQLContext(sc);
        
        
        DataFrame rawData = sqlContext.createDataFrame(documents, createStructType());
        Tokenizer tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words");
        DataFrame wordsData = tokenizer.transform(rawData);
        HashingTF hashingTF = new HashingTF().setNumFeatures(10000).setInputCol("words").setOutputCol("rawfeature"); 
        DataFrame featuredData = hashingTF.transform(wordsData);
        IDF idf =new IDF().setInputCol("rawfeature").setOutputCol("feature");
        DataFrame idfModel = idf.fit(featuredData).transform(featuredData);
        
        JavaRDD<Row> rescaletRDD = idfModel.select("words","feature").toJavaRDD();
        
        JavaPairRDD<Integer, String> resultRDD = rescaletRDD.flatMap(new FlatMapFunction<Row, String>() {
		  public Iterable<String> call(Row arg0) throws Exception {
				String outputString = arg0.get(0).toString();
				String words = outputString.substring(outputString.lastIndexOf("(")+1,outputString.length()-1);
				String feature = arg0.get(1).toString();
				String value = feature.substring(feature.lastIndexOf("[")+1, feature.lastIndexOf("]"));
				String [] word = words.split(",");
				String [] features = value.split(",");
				ArrayList<String> arrayList = new ArrayList<>();
				for(String v : word){
					if(v.length()>=2){
						arrayList.add(v);
					}
				}
				return arrayList;
			}
		}).mapToPair(new PairFunction<String, String, Integer>() {
			public Tuple2<String, Integer> call(String arg0) throws Exception {
				return new Tuple2<String, Integer>(arg0, 1);
			}
		}).reduceByKey(new Function2<Integer, Integer, Integer>() {
			public Integer call(Integer arg0, Integer arg1) throws Exception {
				return arg0 + arg1;
			}
		}).mapToPair(new PairFunction<Tuple2<String,Integer>, Integer, String>() {
			public Tuple2<Integer, String> call(Tuple2<String, Integer> arg0)
					throws Exception {
				return arg0.swap();
			}
		}).sortByKey(false);
        
//        JavaPairRDD<Double, String> resultRDD = rescaletRDD.mapToPair(new PairFunction<Row, Double, String>() {
//			public Tuple2<Double, String> call(Row arg0) throws Exception {
//				String outputString = arg0.get(0).toString();
//				String words = outputString.substring(outputString.lastIndexOf("(")+1,outputString.length()-1);
//				String feature = arg0.get(1).toString();
//				String value = feature.substring(feature.lastIndexOf("[")+1, feature.lastIndexOf("]"));
//				String [] word = words.split(",");
//				String [] features = value.split(",");
//				Double biggest = 0.0;
//				int index = 0;
//				for(int i=0 ;i<features.length ;i++){
//					if(Double.parseDouble(features[i]) > index){
//						biggest = Double.parseDouble(features[i]) ;
//						index = i;
//					}
//				}
//				return new Tuple2<Double, String>(biggest, word[index]);
//			}
//		}).sortByKey(false);
//        
        for(Tuple2<Integer, String> v : resultRDD.take(20)){
        	System.out.println(v);
        }
        
      
    }
    private static StructType createStructType() {
		List<StructField> fields = new ArrayList<>();
	    fields.add(DataTypes.createStructField("text", DataTypes.StringType, true));
	    StructType schema = DataTypes.createStructType(fields);
		return schema;
	}
}
