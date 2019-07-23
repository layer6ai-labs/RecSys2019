package common.xgb;

import common.linalg.MLSparseMatrix;
import common.linalg.MLSparseVector;
import common.utils.MLConcurrentUtils.Async;
import ml.dmlc.xgboost4j.LabeledPoint;
import ml.dmlc.xgboost4j.java.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

public final class MLXGBoost {

	private MLXGBoost() {}	// Prevent instantiation

	public static class MLXGBoostFeature {

		public static class ScoreComparator
				implements Comparator<MLXGBoostFeature> {

			private boolean decreasing;

			public ScoreComparator(final boolean decreasingP) {
				this.decreasing = decreasingP;
			}

			@Override
			public int compare(final MLXGBoostFeature e1,
					final MLXGBoostFeature e2) {
				if (this.decreasing == true) {
					return Double.compare(e2.score, e1.score);
				} else {
					return Double.compare(e1.score, e2.score);
				}
			}
		}

		private String name;
		private double score;

		public MLXGBoostFeature(final String nameP, final double scoreP) {
			this.name = nameP;
			this.score = scoreP;
		}

		public String getName() {
			return this.name;
		}

		public double getScore() {
			return this.score;
		}
	}

	public static MLXGBoostFeature[] analyzeFeatures(final String modelFile,
			final String featureFile) throws Exception {

		Booster model = XGBoost.loadModel(modelFile);

		List<String> temp = new LinkedList<String>();
		try (BufferedReader reader = new BufferedReader(
				new FileReader(featureFile))) {
			String line;
			while ((line = reader.readLine()) != null) {
				temp.add(line);
			}
		}

		// get feature importance scores
		String[] featureNames = new String[temp.size()];
		temp.toArray(featureNames);
		int[] importances = MLXGBoost.getFeatureImportance(model, featureNames);

		// sort features by their importance
		MLXGBoostFeature[] sortedFeatures = new MLXGBoostFeature[featureNames.length];
		for (int i = 0; i < featureNames.length; i++) {
			sortedFeatures[i] = new MLXGBoostFeature(featureNames[i],
					importances[i]);
		}
		Arrays.sort(sortedFeatures, new MLXGBoostFeature.ScoreComparator(true));

		return sortedFeatures;
	}

	public static Async<Booster> asyncModel(final String modelFile) {
		return asyncModel(modelFile, 0);
	}

	public static Async<Booster> asyncModel(final String modelFile,
			final int nthread) {
		// load xgboost model
		final Async<Booster> modelAsync = new Async<Booster>(() -> {
			try {
				Booster bst = XGBoost.loadModel(modelFile);
				if (nthread > 0) {
					bst.setParam("nthread", nthread);
				}
				return bst;
			} catch (XGBoostError e) {
				e.printStackTrace();
				return null;
			}
		}, Booster::dispose);
		return modelAsync;
	}

	public static int[] getFeatureImportance(final Booster model,
			final String[] featNames) throws XGBoostError {

		int[] importances = new int[featNames.length];
		// NOTE: not used feature are dropped here
		Map<String, Integer> importanceMap = model.getFeatureScore((String) null);

		for (Map.Entry<String, Integer> entry : importanceMap.entrySet()) {
			// get index from f0, f1 feature name output from xgboost
			int index = Integer.parseInt(entry.getKey().substring(1));
			importances[index] = entry.getValue();
		}

		return importances;
	}

	public static DMatrix toDMatrix(final MLSparseMatrix matrix)
			throws XGBoostError {

		final int nnz = (int) matrix.getNNZ();
		final int nRows = matrix.getNRows();
		final int nCols = matrix.getNCols();

		long[] rowIndex = new long[nRows + 1];
		int[] indexesFlat = new int[nnz];
		float[] valuesFlat = new float[nnz];

		int cur = 0;
		for (int i = 0; i < nRows; i++) {
			MLSparseVector row = matrix.getRow(i);
			if (row == null) {
				rowIndex[i] = cur;
				continue;
			}
			int[] indexes = row.getIndexes();
			int rowNNZ = indexes.length;
			if (rowNNZ == 0) {
				rowIndex[i] = cur;
				continue;
			}
			float[] values = row.getValues();
			rowIndex[i] = cur;

			for (int j = 0; j < rowNNZ; j++, cur++) {
				indexesFlat[cur] = indexes[j];
				valuesFlat[cur] = values[j];
			}
		}
		rowIndex[nRows] = cur;
		return new DMatrix(rowIndex, indexesFlat, valuesFlat,
				DMatrix.SparseType.CSR, nCols);
	}

	public static String toLIBSVMString(final LabeledPoint vec) {
		float target = vec.label();
		StringBuilder builder = new StringBuilder();
		if (target == (int) target) {
			builder.append((int) target);
		} else {
			builder.append(String.format("%.5f", target));
		}
		for (int i = 0; i < vec.indices().length; i++) {
			float val = vec.values()[i];
			if (val == Math.round(val)) {
				builder.append(" " + (vec.indices()[i]) + ":" + ((int) val));
			} else {
				builder.append(" " + (vec.indices()[i]) + ":"
						+ String.format("%.5f", val));
			}
		}
		return builder.toString();
	}

	/**
	 * Wrapper class holding the performance-vs-iteration of XGBoost training on a single metric.
	 */
	public static class TrainingRecord {
		private float[] scores;
		private float maxScore;
		private int iterAtMaxScore;

		private TrainingRecord(float[] scores, float maxScore, int iterOfMaxScore) {
			this.scores = scores;
			this.maxScore = maxScore; this.iterAtMaxScore = iterOfMaxScore;
		}

		/**
		 * Get the scores at each iteration.
		 * @return float[] of scores.
		 */
		public float[] getScores() {
			return scores;
		}

		/**
		 * Return the maximum score.
		 * @return Maximum score.
		 */
		public float getMaxScore() {
			return maxScore;
		}

		/**
		 * Return the iteration at which the maximum score occurred (useful for choosing when to stop).
		 * @return Iteration at which max score occurred.
		 */
		public int getIterAtMaxScore() {
			return iterAtMaxScore;
		}
	}

	/**
	 * Wrapper class holding the performance-vs-iteration of XGBoost training.
	 */
	public static class BoosterAndTrainingRecord {
		private Booster booster;
		private HashMap<String, TrainingRecord> trainingRecords = new HashMap<>();

		private BoosterAndTrainingRecord(Booster booster, Map<String, ?> watches, float[][] metrics) {
			this.booster = booster;
			int j=0;
			// Load the different evaluation metrics into the training record
		 	for (Map.Entry<String, ?> entry : watches.entrySet()) {

		 		// Find the max score and the iter it is attained at
				float maxScore = Float.NaN;
				int iterAtMaxScore = 0;
				for (int i = 0; i < metrics[j].length; i++) {
					float score = metrics[j][i];
					if (Float.isNaN(maxScore) || score < maxScore) {
						maxScore = score;
						iterAtMaxScore = i;
					}
				}

				trainingRecords.put(entry.getKey(), new TrainingRecord(metrics[j], maxScore, iterAtMaxScore));

		 		j++;
			}
		}

		/**
		 * Get the value of the metric at each iteration, plus the max value and which iteration the maximum was at.
		 * @param metricName Name of the metric as passed into watches for XGBoost.
		 * @return TrainingRecord
		 */
		public TrainingRecord getTrainingRecord(String metricName) {
			return trainingRecords.get(metricName);
		}

		public Booster getBooster() {
			return this.booster;
		}
	}

	/**
	 * Wrapped version of XGBoost that returns an object packaging up the scores.
	 *
	 * @param dtrain DMatrix of training features and labels.
	 * @param params Dictionary of params for XGBoost.
	 * @param round Number of iterations to train for.
	 * @param watches DMatrices to evaluate on (i.e. validation set).
	 * @param obj Can be null.
	 * @param eval Can be null.
	 * @return A BoosterAndTrainingRecord containing the Booster and a TrainingRecord for each entry in watch.
	 * @throws XGBoostError Passed up from XGBoost
	 */
	public static BoosterAndTrainingRecord trainWithScores(
			DMatrix dtrain,
			Map<String, Object> params,
			int round,
			Map<String, DMatrix> watches,
			IObjective obj,
			IEvaluation eval) throws XGBoostError {

		// Data that XGBoost saves the training record into
		float[][] metrics = new float[watches.size()][round];

		Booster booster = XGBoost.train(dtrain, params, round, watches, metrics, obj, eval, 0, null);

		return new BoosterAndTrainingRecord(booster, watches, metrics);
	}

	/**
	 * Train an XGBoost model and record validation performance.
	 *
	 * @param trainingFeaturesAndLabels Path to libSVM file for training features and labels.
	 * @param validationFeaturesAndLabels Path to libSVM file for validation features and labels.
	 * @param params Dictionary of params for XGBoost.
	 * @param round Number of iterations to train for.
	 * @return A BoosterAndTrainingRecord containing the Booster and a TrainingRecord for each entry in watch.
	 * @throws XGBoostError Passed up from XGBoost
	 */
	public static BoosterAndTrainingRecord trainWithScores(
			String trainingFeaturesAndLabels,
			String validationFeaturesAndLabels,
			Map<String, Object> params,
			int round) throws XGBoostError {

		DMatrix training = new DMatrix(trainingFeaturesAndLabels);
		DMatrix validation = new DMatrix(validationFeaturesAndLabels);

		Map<String, DMatrix> watches = new HashMap<>();
		watches.put("training", training);
		watches.put("validation", validation);

		return trainWithScores(training, params, round, watches, null, null);

	}
}
