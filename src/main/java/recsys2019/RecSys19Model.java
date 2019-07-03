package recsys2019;
import common.eval.MLEvaluator;
import common.eval.MLEvaluatorAUC;
import common.eval.MLEvaluatorMRR;
import common.linalg.FloatElement;
import common.linalg.MLSparseMatrix;
import common.linalg.MLSparseMatrixAOO;
import common.linalg.MLSparseMatrixFlat;
import common.linalg.MLSparseVector;
import common.utils.MLConcurrentUtils;
import common.utils.MLIOUtils;
import common.utils.MLRandomUtils;
import common.utils.MLTimer;
import common.xgb.MLXGBoost;
import common.xgb.XGBEvaluator;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.IEvaluation;
import ml.dmlc.xgboost4j.java.XGBoost;
import recsys2019.RecSys19Data.SessionFeature;
import recsys2019.RecSys19FeatureExtractor.SessionInstance;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class RecSys19Model {

    public static class RecSys19Config {
        public long trainDelta = 100_000;
        public boolean removeTrain = true;
        public boolean removeValid = true;
        public int nTrainZeros = 20;
        public float sampleTargetProb = 0.2f;
    }

    private static MLTimer timer;

    static {
        timer = new MLTimer("RecSys19Model");
        timer.tic();
    }

    private RecSys19Data data;
    private RecSys19Config config;
    private RecSys19FeatureExtractor featExtractor;
    private MLSparseMatrix validTargets;
    private MLEvaluatorMRR evaluator;

    public RecSys19Model(final RecSys19Data dataP, final RecSys19Config configP) throws Exception {
        this.data = dataP;
        this.config = configP;
        this.data.trainEventIndexes = getTrainIndexes(this.data, this.config, null);
        this.validTargets = createValidMatrix(this.data);
        this.evaluator = new MLEvaluatorMRR();
        this.featExtractor = new RecSys19FeatureExtractor(this.data, this.config);
    }

    public static MLSparseMatrix createValidMatrix(final RecSys19Data data) {
        int[] indexes = new int[data.validEventIndexes.length];
        float[] values = new float[indexes.length];
        Arrays.fill(values, 1.0f);
        for (int i = 0; i < indexes.length; i++) {
            int targetIndex = data.validEventIndexes[i];
            indexes[i] = data.referenceItems[targetIndex];
        }
        return new MLSparseMatrixFlat(indexes, values, data.itemIdToIndex.size());
    }

    public static int[] getTrainIndexes(final RecSys19Data data, final RecSys19Config config, final String outFile) throws Exception {
        Map<Integer, String> indexToUser = data.sessionFeatures.get(SessionFeature.user_id).getIndexToCat();
        Map<Integer, String> indexToSession = data.sessionFeatures.get(SessionFeature.session_id).getIndexToCat();
        final long TRAIN_END = RecSys19Data.VALID_SPLIT_START;
        final long TRAIN_START = TRAIN_END - config.trainDelta;
        final int clickAction = RecSys19Helper.getActionIndex(RecSys19Data.CLICKOUT_ITEM_ACTION, data);
        List<Integer> trainIndexes = new ArrayList();
        BufferedWriter writer = null;
        if (outFile != null) {
            writer = new BufferedWriter(new FileWriter(outFile));
        }
        Random random = new Random(1);
        for (Map.Entry<Integer, Set<Integer>> entry : data.trainToSessionStart.entrySet()) {
            for (int startIndex : entry.getValue()) {
                if (data.timeStamps[startIndex] < TRAIN_START || data.timeStamps[startIndex] > TRAIN_END) {
                    continue;
                }
                int sessionIndex = RecSys19Helper.getIndex(startIndex, SessionFeature.session_id, data);
                int curIndex = startIndex - 1;
                List<Integer> allClicks = new LinkedList();
                while (true) {
                    curIndex++;
                    if (sessionIndex != RecSys19Helper.getIndex(curIndex, SessionFeature.session_id, data)) {
                        break;
                    }
                    int nextSessionIndex = RecSys19Helper.getIndex(curIndex + 1, SessionFeature.session_id, data);
                    int curAction = RecSys19Helper.getIndex(curIndex, SessionFeature.action_type, data);
                    if (curAction != clickAction) {
                        continue;
                    }
                    int clickedItem = data.referenceItems[curIndex];
                    if (clickedItem < 0) {
                        continue;
                    }
                    boolean found = false;
                    int[] impressions = data.impressions[curIndex];
                    for (int i = 0; i < impressions.length; i++) {
                        if (clickedItem == impressions[i]) {
                            found = true;
                            break;
                        }
                    }
                    if (found == false || RecSys19Helper.isValidIndex(curIndex, data) == true) {
                        continue;
                    }
                    if (sessionIndex == nextSessionIndex) {
                        allClicks.add(curIndex);
                    }
                    if (sessionIndex != nextSessionIndex || random.nextFloat() < config.sampleTargetProb) {
                        trainIndexes.add(curIndex);
                        if (writer != null) {
                            String userId = indexToUser.get(RecSys19Helper.getIndex(curIndex, SessionFeature.user_id, data));
                            String sessionId = indexToSession.get(RecSys19Helper.getIndex(curIndex, SessionFeature.session_id, data));
                            int step = (int) RecSys19Helper.getValue(curIndex, SessionFeature.step, data);
                            long timeStamp = data.timeStamps[curIndex];
                            writer.write(userId + "," + sessionId + "," + timeStamp + "," + step + "\n");
                        }
                    }
                }
            }
        }
        if (writer != null) {
            writer.close();
        }
        int[] trainEventIndexes = new int[trainIndexes.size()];
        for (int i = 0; i < trainEventIndexes.length; i++) {
            trainEventIndexes[i] = trainIndexes.get(i);
        }
        Arrays.sort(trainEventIndexes);
        timer.toc("nTrainSessions " + trainEventIndexes.length);
        return trainEventIndexes;
    }

    public void extractXGBModel(final String trainFile, final String validFile, final boolean combine) {
        int nTrain = this.data.trainEventIndexes.length;
        int nValid = this.data.validEventIndexes.length;
        try (BufferedWriter trainWriter = new BufferedWriter(new FileWriter(trainFile));
             BufferedWriter trainGroupWriter = new BufferedWriter(new FileWriter(trainFile + ".gr"));
             BufferedWriter validWriter = new BufferedWriter(new FileWriter(validFile));
             BufferedWriter validGroupWriter = new BufferedWriter(new FileWriter(validFile + ".gr"))) {
            AtomicInteger counter = new AtomicInteger(0);
            AtomicInteger counterPrint = new AtomicInteger(0);
            IntStream.range(0, nTrain + nValid).parallel().forEach(index -> {
                int count = counter.incrementAndGet();
                if (count % 50_000 == 0) {
                    timer.tocLoop("extractXGBModel", count);
                }
                int targetIndex;
                boolean isTrain;
                if (index < nTrain) {
                    targetIndex = this.data.trainEventIndexes[index];
                    isTrain = true;
                } else {
                    targetIndex = this.data.validEventIndexes[index - nTrain];
                    isTrain = false;
                }
                SessionInstance[] instances = this.featExtractor.extractFeatures(targetIndex);
                if (instances.length > 0 && counterPrint.incrementAndGet() == 1) {
                    System.out.println("nFeats = " + instances[0].features.getLength());
                }
                if (isTrain == true || combine == true) {
                    StringBuilder builder = new StringBuilder();
                    MLRandomUtils.shuffle(instances, new Random(index));
                    int sampleCount = 0;
                    for (SessionInstance instance : instances) {
                        if (instance.target == 1) {
                            builder.append(instance.target + instance.features.toLIBSVMString(0) + "\n");
                        } else if (sampleCount < this.config.nTrainZeros) {
                            builder.append(instance.target + instance.features.toLIBSVMString(0) + "\n");
                            sampleCount++;
                        }
                    }
                    synchronized (trainWriter) {
                        try {
                            trainWriter.write(builder.toString());
                            trainGroupWriter.write((sampleCount + 1) + "\n");
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
                if (isTrain == false) {
                    StringBuilder builder = new StringBuilder();
                    for (SessionInstance instance : instances) {
                        builder.append(instance.target + instance.features.toLIBSVMString(0) + "\n");
                    }
                    synchronized (validWriter) {
                        try {
                            validWriter.write(builder.toString());
                            validGroupWriter.write(instances.length + "\n");
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }

            });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void trainXGBModel(final String trainFile, final String validFile, final String modelPath, final String modelVersion) throws Exception {
        int rounds;
        int earlyStoppingRounds;
        System.out.printf("Loading '%s'...\n", trainFile);
        DMatrix trainData = new DMatrix(trainFile);
        System.out.printf("Loading '%s'...\n", validFile);
        DMatrix validData = new DMatrix(validFile);
        Map<String, Object> params = new HashMap<>();
        params.put("booster", "gbtree");
        params.put("eta", 0.1);
        params.put("gamma", 0);
        params.put("min_child_weight", 1);
        params.put("max_depth", 10);
        params.put("subsample", 1);
        params.put("colsample_bynode", 0.8);
        params.put("seed", 3);
        params.put("scale_pos_weight", 1);
        params.put("objective", "binary:logistic");
        params.put("base_score", 0.1);
        params.put("use_buffer", 1);
        params.put("verbosity", 2);
        params.put("predictor", "cpu_predictor");
        params.put("eval_metric", "auc");
        params.put("maximize_evaluation_metrics", true);
        if (modelVersion.equals("1")) { // hist + no regularization
            rounds = 1_000;
            earlyStoppingRounds = 20;
            params.put("lambda", 1);
            params.put("alpha", 0);
            params.put("tree_method", "hist");
        } else if (modelVersion.equals("2")) { // exact + regularization
            rounds = 8_000;
            earlyStoppingRounds = 50;
            params.put("lambda", 4000);
            params.put("alpha", 10);
            params.put("tree_method", "exact");
        } else {
            throw new Exception("Invalid modelVersion given!");
        }
        System.out.println("Parameters:");
        for (String key : params.keySet()) {
            System.out.printf(" %s = %s\n", key, params.get(key));
        }
        HashMap<String, DMatrix> watches = new HashMap<>();
        watches.put("valid", validData);
        IEvaluation eval = new XGBEvaluator(new MLEvaluatorAUC(false), false, 2, false);
        Booster booster = XGBoost.train(trainData, params, rounds, watches, null,
                    null, eval, earlyStoppingRounds, null);
        booster.saveModel(modelPath + "model.bin");
    }

    public void validateXGBModel(final String xgbModel, final String predFile, final boolean coldStartOnly) throws Exception {
        final MLConcurrentUtils.Async<Booster> xgbModelFactory = MLXGBoost.asyncModel(xgbModel);
        AtomicInteger counter = new AtomicInteger(0);
        AtomicInteger counterPrint = new AtomicInteger(0);
        FloatElement[][] preds = new FloatElement[this.data.validEventIndexes.length][];
        FloatElement[][] predsUnsorted = new FloatElement[this.data.validEventIndexes.length][];
        IntStream.range(0, this.data.validEventIndexes.length).parallel().forEach(index -> {
            int count = counter.incrementAndGet();
            if (count % 50_000 == 0) {
                timer.tocLoop("validateXGBModel", count);
            }
            int targetIndex = this.data.validEventIndexes[index];
            if (coldStartOnly == true) {
                int step = (int) RecSys19Helper.getValue(targetIndex, SessionFeature.step, this.data);
                int userIndex = RecSys19Helper.getIndex(targetIndex, SessionFeature.user_id, this.data);
                if (step != 1 || this.data.userToSessionStart.get(userIndex).size() != 1) {
                    return;
                }
            }
            SessionInstance[] instances = this.featExtractor.extractFeatures(targetIndex);
            if (instances.length > 0 && counterPrint.incrementAndGet() == 1) {
                System.out.println("nFeats = " + instances[0].features.getLength());
            }
            MLSparseVector[] feats = new MLSparseVector[instances.length];
            for (int i = 0; i < instances.length; i++) {
                feats[i] = instances[i].features;
            }
            FloatElement[] pred = new FloatElement[feats.length];
            DMatrix xgbMat = null;
            try {
                xgbMat = MLXGBoost.toDMatrix(new MLSparseMatrixAOO(feats, feats[0].getLength()));
                float[][] xgbPreds = xgbModelFactory.get().predict(xgbMat);
                for (int i = 0; i < feats.length; i++) {
                    pred[i] = new FloatElement(instances[i].itemIndex, xgbPreds[i][0]);
                }
                if (predFile != null) {
                    predsUnsorted[index] = pred.clone();
                }
                Arrays.sort(pred, new FloatElement.ValueComparator(true));
                preds[index] = pred;
            } catch (Exception e) {
                e.printStackTrace();
                throw new RuntimeException("xgb failed");
            } finally {
                xgbMat.dispose();
            }
        });
        if (predFile != null) {
            MLIOUtils.writeObjectToFile(predsUnsorted, predFile);
        }
        timer.tocLoop("validateXGBModel", counter.get());
        timer.toc(this.evaluator.evaluate(this.validTargets, preds).toString());
    }

    public void submitXGBModel(final String xgbModel, final String outFile, final String predFile) {
        final MLConcurrentUtils.Async<Booster> xgbModelFactory = MLXGBoost.asyncModel(xgbModel);
        Map<Integer, String> indexToUser = this.data.sessionFeatures.get(SessionFeature.user_id).getIndexToCat();
        Map<Integer, String> indexToSession = this.data.sessionFeatures.get(SessionFeature.session_id).getIndexToCat();
        Map<Integer, Integer> indexToItem = this.data.getIndexToItemId();
        AtomicInteger counter = new AtomicInteger(0);
        AtomicInteger counterPrint = new AtomicInteger(0);
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outFile))) {
            writer.write("user_id,session_id,timestamp,step,item_recommendations\n");
            FloatElement[][] preds = new FloatElement[this.data.testEventIndexes.length][];
            IntStream.range(0, this.data.testEventIndexes.length).parallel().forEach(index -> {
                int count = counter.incrementAndGet();
                if (count % 50_000 == 0) {
                    timer.tocLoop("submitXGBModel", count);
                }
                int targetIndex = this.data.testEventIndexes[index];
                SessionInstance[] instances = this.featExtractor.extractFeatures(targetIndex);
                if (instances.length > 0 && counterPrint.incrementAndGet() == 1) {
                    System.out.println("nFeats = " + instances[0].features.getLength());
                }
                MLSparseVector[] feats = new MLSparseVector[instances.length];
                for (int i = 0; i < instances.length; i++) {
                    feats[i] = instances[i].features;
                }
                FloatElement[] pred = new FloatElement[feats.length];
                DMatrix xgbMat = null;
                try {
                    xgbMat = MLXGBoost.toDMatrix(new MLSparseMatrixAOO(feats, feats[0].getLength()));
                    float[][] xgbPreds = xgbModelFactory.get().predict(xgbMat);
                    for (int i = 0; i < feats.length; i++) {
                        pred[i] = new FloatElement(indexToItem.get(instances[i].itemIndex), xgbPreds[i][0]);
                    }
                    if (predFile != null) {
                        preds[index] = pred.clone();
                    }
                    Arrays.sort(pred, new FloatElement.ValueComparator(true));
                } catch (Exception e) {
                    e.printStackTrace();
                    throw new RuntimeException("xgb failed");
                } finally {
                    xgbMat.dispose();
                }
                String userId = indexToUser.get(RecSys19Helper.getIndex(targetIndex, SessionFeature.user_id, this.data));
                String sessionId = indexToSession.get(RecSys19Helper.getIndex(targetIndex, SessionFeature.session_id, this.data));
                long timeStamp = this.data.timeStamps[targetIndex];
                int step = (int) RecSys19Helper.getValue(targetIndex, SessionFeature.step, this.data);
                StringBuilder builder = new StringBuilder();
                builder.append(userId + "," + sessionId + "," + timeStamp + "," + step + ",");
                for (FloatElement element : pred) {
                    builder.append(element.getIndex() + " ");
                }
                synchronized (writer) {
                    try {
                        writer.write(builder.toString().trim() + "\n");
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            });
            if (predFile != null) {
                MLIOUtils.writeObjectToFile(preds, predFile);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        timer.tocLoop("submitXGBModel", counter.get());
    }

    public static void main(final String[] args) {
        try {
            String dataPath = args[0];
            String outPath = args[1];
            String modelVersion = args[2];
            String runMode = args[3];
            if (!new File(dataPath).exists()) {
                throw new Exception("Invalid dataPath given!");
            }
            if (!new File(outPath).exists()) {
                throw new Exception("Invalid outPath given!");
            }
            if (!(modelVersion.equals("1") || modelVersion.equals("2"))) {
                throw new Exception("Invalid modelVersion given!");
            }
            if (!(runMode.equals("extract") || runMode.equals("train") ||
                    runMode.equals("validate") || runMode.equals("submit"))) {
                throw new Exception("Invalid runMode given!");
            }
            RecSys19Data data = MLIOUtils.readObjectFromFile(outPath + "data.parsed", RecSys19Data.class);
            timer.toc("data loaded");
            RecSys19Config config = new RecSys19Config();
            config.sampleTargetProb = 0.2f;
            config.trainDelta = 1_000_000;
            if (runMode.equals("validate") || runMode.equals("submit")) {
                config.removeTrain = false;
            }
            RecSys19Model model = new RecSys19Model(data, config);
            String removeTrainStr = "_removeTrain=" + (config.removeTrain ? "1":"0");
            String removeValidStr = "_removeValid=" + (config.removeValid ? "1":"0");
            String modelFile = outPath + "model.bin";
            String predValidFile = outPath + "xgb_valid" + removeTrainStr + removeValidStr + ".preds";
            String predTestFile = outPath + "xgb_test" + removeTrainStr + removeValidStr + ".preds";
            String submitFile = outPath + "submit.csv";
            if (runMode.equals("extract")) {
                model.extractXGBModel(outPath + "trainXGB", outPath + "validXGB", false);
            } else if (runMode.equals("train")) {
                model.trainXGBModel(outPath + "trainXGB", outPath + "validXGB", outPath, modelVersion);
            } else if (runMode.equals("validate")) {
                model.validateXGBModel(modelFile, predValidFile, false);
            } else if (runMode.equals("submit")) {
                model.submitXGBModel(modelFile, submitFile, predTestFile);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}