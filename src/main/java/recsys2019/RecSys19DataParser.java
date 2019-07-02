package recsys2019;
import common.feature.MLFeatureTransform;
import common.feature.MLSparseFeature;
import common.linalg.MLSparseMatrixAOO;
import common.linalg.MLSparseMatrixFlat;
import common.utils.MLIOUtils;
import common.utils.MLTimer;
import recsys2019.RecSys19Data.ItemFeature;
import recsys2019.RecSys19Data.SessionFeature;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

public class RecSys19DataParser {
    public static final int N_SESSION_ROWS = 15932993 + 3782336 - 2;
    public static final int N_ITEM_ROWS = 927997;
    private static MLTimer timer;

    static {
        timer = new MLTimer("RecSys19DataParser");
        timer.tic();
    }

    public RecSys19Data data;
    public int curItemIndex;
    public int curSessionIndex;

    public RecSys19DataParser() {
        this.data = new RecSys19Data();
    }

    public void parseItemData(final String file) throws Exception {
        this.data.itemIdToIndex = new HashMap();
        String header = Files.lines(Paths.get(file)).iterator().next();
        String[] headerSplit = header.split(",");
        this.data.itemFeatures = new HashMap();
        this.data.itemIdToIndex = new HashMap();
        this.curItemIndex = 0;
        MLSparseFeature feature = new MLSparseFeature(N_ITEM_ROWS, null,
                        new MLFeatureTransform[]{new MLFeatureTransform.ColSelectorTransform(1_000)}, MLSparseMatrixAOO.class);
        this.data.itemFeatures.put(ItemFeature.properties, feature);
        int idIndex = ItemFeature.getColumnIndex(headerSplit, ItemFeature.item_id);
        int propIndex = ItemFeature.getColumnIndex(headerSplit, ItemFeature.properties);
        try (BufferedReader fileReader = new BufferedReader(new FileReader(file))) {
            fileReader.readLine();
            String line;
            while ((line = fileReader.readLine()) != null) {
                String[] split = line.split(",");
                if (split.length != headerSplit.length) {
                    throw new Exception("split length doesn't match");
                }
                int itemId = Integer.parseInt(split[idIndex]);
                this.data.itemIdToIndex.put(itemId, this.curItemIndex);
                String[] props = split[propIndex].split("\\|");
                this.data.itemFeatures.get(ItemFeature.properties).addRow(this.curItemIndex, props);
                this.curItemIndex++;
                if (this.curItemIndex % 100_000 == 0) {
                    timer.tocLoop("parseItemData", this.curItemIndex);
                }
            }
        }
        timer.tocLoop("parseItemData", this.curItemIndex);
        for (Map.Entry<ItemFeature, MLSparseFeature> entry : this.data.itemFeatures.entrySet()) {
            entry.getValue().finalizeFeature(true);
            timer.toc("parseItemData nCols retained " + entry.getValue().getFeatMatrixTransformed().getNCols());
        }
    }

    private void initSessionFeatures() {
        this.data.sessionFeatures = new HashMap();
        for (SessionFeature featureName : SessionFeature.values()) {
            switch (featureName) {
                case user_id: {
                    MLSparseFeature feature = new MLSparseFeature(N_SESSION_ROWS, null,
                                    null, MLSparseMatrixFlat.class);
                    this.data.sessionFeatures.put(featureName, feature);
                    break;
                }
                case session_id: {
                    MLSparseFeature feature = new MLSparseFeature(N_SESSION_ROWS, null,
                                    null, MLSparseMatrixFlat.class);
                    this.data.sessionFeatures.put(featureName, feature);
                    break;
                }
                case timestamp: {
                    break;
                }
                case step: {
                    MLSparseFeature feature = new MLSparseFeature(N_SESSION_ROWS, null,
                                    null, MLSparseMatrixFlat.class);
                    this.data.sessionFeatures.put(featureName, feature);
                    break;
                }
                case action_type: {
                    MLSparseFeature feature = new MLSparseFeature(N_SESSION_ROWS, null,
                                    null, MLSparseMatrixFlat.class);
                    this.data.sessionFeatures.put(featureName, feature);
                    break;
                }
                case reference_filter: {
                    MLSparseFeature feature = new MLSparseFeature(N_SESSION_ROWS, null,
                                    null, MLSparseMatrixFlat.class);
                    this.data.sessionFeatures.put(featureName, feature);
                    break;
                }
                case reference_sort_order: {
                    MLSparseFeature feature = new MLSparseFeature(N_SESSION_ROWS, null,
                                    null, MLSparseMatrixFlat.class);
                    this.data.sessionFeatures.put(featureName, feature);
                    break;
                }
                case reference_search_dest: {
                    MLSparseFeature feature = new MLSparseFeature(N_SESSION_ROWS, null,
                                    null, MLSparseMatrixFlat.class);
                    this.data.sessionFeatures.put(featureName, feature);
                    break;
                }
                case reference_search_poi: {
                    MLSparseFeature feature = new MLSparseFeature(N_SESSION_ROWS, null,
                                    null, MLSparseMatrixFlat.class);
                    this.data.sessionFeatures.put(featureName, feature);
                    break;
                }
                case platform: {
                    MLSparseFeature feature = new MLSparseFeature(N_SESSION_ROWS, null,
                                    null, MLSparseMatrixFlat.class);
                    this.data.sessionFeatures.put(featureName, feature);
                    break;
                }
                case city: {
                    MLSparseFeature feature = new MLSparseFeature(N_SESSION_ROWS, null,
                                    null, MLSparseMatrixFlat.class);
                    this.data.sessionFeatures.put(featureName, feature);
                    break;
                }
                case device: {
                    MLSparseFeature feature = new MLSparseFeature(N_SESSION_ROWS, null,
                                    null, MLSparseMatrixFlat.class);
                    this.data.sessionFeatures.put(featureName, feature);
                    break;
                }
                case current_filters: {
                    MLSparseFeature feature = new MLSparseFeature(N_SESSION_ROWS, null,
                                    null, MLSparseMatrixAOO.class,
                                    this.data.itemFeatures.get(ItemFeature.properties));
                    this.data.sessionFeatures.put(featureName, feature);
                    break;
                }
                case impressions: {
                    break;
                }
                case prices: {
                    break;
                }
            }
        }
    }

    public void createSplit(final String outFile) throws Exception {
        final int clickAction = RecSys19Helper.getActionIndex(RecSys19Data.CLICKOUT_ITEM_ACTION, this.data);
        Map<Integer, String> indexToUser = this.data.sessionFeatures.get(SessionFeature.user_id).getIndexToCat();
        Map<Integer, String> indexToSession = this.data.sessionFeatures.get(SessionFeature.session_id).getIndexToCat();
        List<Integer> validList = new ArrayList();
        int count = 0;
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outFile))) {
            for (Map.Entry<Integer, Set<Integer>> entry : this.data.trainToSessionStart.entrySet()) {
                count++;
                if (count % 100_000 == 0) {
                    timer.tocLoop("createSplit", count);
                }
                for (int startIndex : entry.getValue()) {
                    if (this.data.timeStamps[startIndex] < RecSys19Data.VALID_SPLIT_START
                            || this.data.timeStamps[startIndex] > RecSys19Data.VALID_SPLIT_END) {
                        continue;
                    }
                    int sessionIndex = RecSys19Helper.getIndex(startIndex, SessionFeature.session_id, this.data);
                    int curIndex = startIndex;
                    while (true) {
                        int nextSessionIndex = RecSys19Helper.getIndex(curIndex + 1, SessionFeature.session_id, this.data);
                        if (sessionIndex != nextSessionIndex) {
                            int curAction = RecSys19Helper.getIndex(curIndex, SessionFeature.action_type, this.data);
                            if (curAction == clickAction) {
                                int clickedItem = this.data.referenceItems[curIndex];
                                if (clickedItem < 0) {
                                    break;
                                }
                                boolean found = false;
                                for (int i = 0; i < this.data.impressions[curIndex].length; i++) {
                                    if (clickedItem == this.data.impressions[curIndex][i]) {
                                        found = true;
                                        break;
                                    }
                                }
                                if (found == false) {
                                    break;
                                }
                                validList.add(curIndex);
                                String userId = indexToUser.get(RecSys19Helper.getIndex(curIndex, SessionFeature.user_id, this.data));
                                String sessionId = indexToSession.get(RecSys19Helper.getIndex(curIndex, SessionFeature.session_id, this.data));
                                int step = (int) RecSys19Helper.getValue(curIndex, SessionFeature.step, this.data);
                                long timeStamp = this.data.timeStamps[curIndex];
                                writer.write(userId + "," + sessionId + "," + timeStamp + "," + step + "\n");
                            }
                            break;
                        }
                        curIndex++;
                    }
                }
            }
        }
        this.data.validEventIndexes = new int[validList.size()];
        for (int i = 0; i < this.data.validEventIndexes.length; i++) {
            this.data.validEventIndexes[i] = validList.get(i);
        }
        Arrays.sort(this.data.validEventIndexes);
        timer.toc("nValidSessions " + this.data.validEventIndexes.length);
    }

    public void parseSessionData(final String trainFile, final String testFile) throws Exception {
        this.initSessionFeatures();
        this.data.userToSessionStart = new HashMap();
        this.data.trainToSessionStart = new HashMap();
        this.data.testToSessionStart = new HashMap();
        this.data.referenceItems = new int[N_SESSION_ROWS];
        Arrays.fill(this.data.referenceItems, -1);
        this.data.impressions = new int[N_SESSION_ROWS][];
        this.data.prices = new int[N_SESSION_ROWS][];
        this.data.timeStamps = new long[N_SESSION_ROWS];
        this.curSessionIndex = -1;
        this.parseSessionData(trainFile, true);
        this.parseSessionData(testFile, false);
        for (Map.Entry<SessionFeature, MLSparseFeature> entry :
                this.data.sessionFeatures.entrySet()) {
            entry.getValue().finalizeFeature(true);
        }
    }

    private void parseSessionData(final String file, final boolean isTrain) throws Exception {
        List<Integer> testIndexes = new ArrayList();
        int sessionCount = 0;
        long minTime = 0;
        long maxTime = 0;
        try (BufferedReader fileReader = new BufferedReader(new FileReader(file))) {
            String[] headerSplit = fileReader.readLine().split(",");
            int userIndexCSV = SessionFeature.getColumnIndex(headerSplit, SessionFeature.user_id);
            int sessIndexCSV = SessionFeature.getColumnIndex(headerSplit, SessionFeature.session_id);
            int timeIndexCSV = SessionFeature.getColumnIndex(headerSplit, SessionFeature.timestamp);
            int stepIndexCSV = SessionFeature.getColumnIndex(headerSplit, SessionFeature.step);
            int actionIndexCSV = SessionFeature.getColumnIndex(headerSplit, SessionFeature.action_type);
            int referenceIndexCSV = SessionFeature.getColumnIndex(headerSplit, SessionFeature.reference);
            int platformIndexCSV = SessionFeature.getColumnIndex(headerSplit, SessionFeature.platform);
            int cityIndexCSV = SessionFeature.getColumnIndex(headerSplit, SessionFeature.city);
            int deviceIndexCSV = SessionFeature.getColumnIndex(headerSplit, SessionFeature.device);
            int filterIndexCSV = SessionFeature.getColumnIndex(headerSplit, SessionFeature.current_filters);
            int impressIndexCSV = SessionFeature.getColumnIndex(headerSplit, SessionFeature.impressions);
            int priceIndexCSV = SessionFeature.getColumnIndex(headerSplit, SessionFeature.prices);
            int curSession = -1;
            String line;
            while ((line = fileReader.readLine()) != null) {
                this.curSessionIndex++;
                if (this.curSessionIndex % 1_000_000 == 0) {
                    timer.tocLoop("parseSessionData", this.curSessionIndex);
                }
                String[] split = splitWithQuotes(line);
                if (split.length != headerSplit.length) {
                    throw new Exception("split length doesn't match");
                }
                this.data.sessionFeatures.get(SessionFeature.user_id).addRow(this.curSessionIndex, split[userIndexCSV]);
                this.data.sessionFeatures.get(SessionFeature.session_id).addRow(this.curSessionIndex, split[sessIndexCSV]);
                this.data.timeStamps[this.curSessionIndex] = Long.parseLong(split[timeIndexCSV]);
                if (curSession < 0) {
                    minTime = this.data.timeStamps[this.curSessionIndex];
                    maxTime = this.data.timeStamps[this.curSessionIndex];
                } else {
                    if (minTime > this.data.timeStamps[this.curSessionIndex]) {
                        minTime = this.data.timeStamps[this.curSessionIndex];
                    }
                    if (maxTime < this.data.timeStamps[this.curSessionIndex]) {
                        maxTime = this.data.timeStamps[this.curSessionIndex];
                    }
                }
                int step = Integer.parseInt(split[stepIndexCSV]);
                this.data.sessionFeatures.get(SessionFeature.step).addRow(this.curSessionIndex, step);
                if (split[actionIndexCSV].length() > 0) {
                    this.data.sessionFeatures.get(SessionFeature.action_type).addRow(this.curSessionIndex, split[actionIndexCSV]);
                }
                if (split[platformIndexCSV].length() > 0) {
                    this.data.sessionFeatures.get(SessionFeature.platform).addRow(this.curSessionIndex, split[platformIndexCSV]);
                }
                if (split[cityIndexCSV].length() > 0) {
                    this.data.sessionFeatures.get(SessionFeature.city).addRow(this.curSessionIndex, split[cityIndexCSV]);
                }
                if (split[deviceIndexCSV].length() > 0) {
                    this.data.sessionFeatures.get(SessionFeature.device).addRow(this.curSessionIndex, split[deviceIndexCSV]);
                }
                if (split[filterIndexCSV].length() > 0) {
                    this.data.sessionFeatures.get(SessionFeature.current_filters).addRow(this.curSessionIndex, split[filterIndexCSV].split("\\|"));
                }
                int userIndex = RecSys19Helper.getIndex(this.curSessionIndex, SessionFeature.user_id, this.data);
                int sessionIndex = RecSys19Helper.getIndex(this.curSessionIndex, SessionFeature.session_id, this.data);
                if (curSession < 0 || curSession != sessionIndex) {
                    sessionCount++;
                    if (isTrain == true) {
                        addToSetMap(this.data.trainToSessionStart, sessionIndex, this.curSessionIndex);
                    } else {
                        addToSetMap(this.data.testToSessionStart, sessionIndex, this.curSessionIndex);
                    }
                    addToSetMap(this.data.userToSessionStart, userIndex, this.curSessionIndex);
                    curSession = sessionIndex;
                }
                if (split[impressIndexCSV].length() > 0) {
                    String[] impressions = split[impressIndexCSV].split("\\|");
                    String[] prices = split[priceIndexCSV].split("\\|");
                    this.data.impressions[this.curSessionIndex] = new int[impressions.length];
                    this.data.prices[this.curSessionIndex] = new int[prices.length];
                    for (int i = 0; i < impressions.length; i++) {
                        int itemId = Integer.parseInt(impressions[i]);
                        Integer itemIndex = this.data.itemIdToIndex.get(itemId);
                        if (itemIndex == null) {
                            this.data.itemIdToIndex.put(itemId, this.curItemIndex);
                            itemIndex = this.curItemIndex;
                            this.curItemIndex++;
                        }
                        this.data.impressions[this.curSessionIndex][i] = itemIndex;
                        this.data.prices[this.curSessionIndex][i] = Integer.parseInt(prices[i]);
                    }
                }
                String action = split[actionIndexCSV];
                if (action.equals(RecSys19Data.CLICKOUT_ITEM_ACTION) ||
                        action.equals(RecSys19Data.INTERACTION_ITEM_RATING_ACTION) ||
                        action.equals(RecSys19Data.INTERACTION_ITEM_INFO_ACTION) ||
                        action.equals(RecSys19Data.INTERACTION_ITEM_IMAGE_ACTION) ||
                        action.equals(RecSys19Data.INTERACTION_ITEM_DEALS_ACTION) ||
                        action.equals(RecSys19Data.SEARCH_FOR_ITEM_ACTION)) {
                    if (split[referenceIndexCSV].length() > 0 && split[referenceIndexCSV].equals("unknown") == false) {
                        try {
                            Integer itemIndex = this.data.itemIdToIndex.get(Integer.parseInt(split[referenceIndexCSV]));
                            if (itemIndex != null) {
                                this.data.referenceItems[this.curSessionIndex] = itemIndex;
                            }
                        } catch (Exception e) {
                            System.out.println("failed " + line);
                        }
                    }
                } else {
                    if (action.equals(RecSys19Data.CHANGE_OF_SORT_ORDER_ACTION)) {
                        this.data.sessionFeatures.get(SessionFeature.reference_sort_order).addRow(this.curSessionIndex, split[referenceIndexCSV]);
                    } else if (action.equals(RecSys19Data.FILTER_SELECTION_ACTION)) {
                        this.data.sessionFeatures.get(SessionFeature.reference_filter).addRow(this.curSessionIndex, split[referenceIndexCSV]);
                    } else if (action.equals(RecSys19Data.SEARCH_FOR_DESTINATION_ACTION)) {
                        this.data.sessionFeatures.get(SessionFeature.reference_search_dest).addRow(this.curSessionIndex, split[referenceIndexCSV]);
                    } else if (action.equals(RecSys19Data.SEARCH_FOR_POI_ACTION)) {
                        this.data.sessionFeatures.get(SessionFeature.reference_search_poi).addRow(this.curSessionIndex, split[referenceIndexCSV]);
                    } else {
                        throw new IllegalStateException("unknown action " + action);
                    }
                }
                if (isTrain == false && action.equals(RecSys19Data.CLICKOUT_ITEM_ACTION) == true && this.data.referenceItems[this.curSessionIndex] < 0) {
                    testIndexes.add(this.curSessionIndex);
                }
            }
        }
        timer.toc("item count " + this.curItemIndex);
        timer.toc("session count " + sessionCount);
        timer.toc("cur session index " + this.curSessionIndex);
        timer.toc("range " + minTime + " - " + maxTime);
        if (isTrain == false) {
            this.data.testEventIndexes = new int[testIndexes.size()];
            for (int i = 0; i < testIndexes.size(); i++) {
                this.data.testEventIndexes[i] = testIndexes.get(i);
            }
            Arrays.sort(this.data.testEventIndexes);
            timer.toc("test index count " + this.data.testEventIndexes.length);
        }
    }

    private static void addToSetMap(final Map<Integer, Set<Integer>> map, final int key, final int value) {
        Set<Integer> indexes = map.get(key);
        if (indexes == null) {
            indexes = new TreeSet();
            map.put(key, indexes);
        }
        indexes.add(value);
    }

    public static String[] splitWithQuotes(final String input) {
        StringBuilder builder = new StringBuilder(input);
        boolean inQuotes = false;
        for (int i = 0; i < builder.length(); i++) {
            char currentChar = builder.charAt(i);
            if (currentChar == '"') {
                inQuotes = !inQuotes;
            }
            if (currentChar == ',' && inQuotes == true) {
                builder.setCharAt(i, ' ');
            }
        }
        return builder.toString().split(",", -1);
    }

    public static void main(final String[] args) {
        try {
            String dataPath = args[0];
            String outPath = args[1];
            if (!new File(dataPath).exists()) {
                throw new Exception("Invalid dataPath given!");
            }
            if (!new File(outPath).exists()) {
                throw new Exception("Invalid outPath given!");
            }
            RecSys19DataParser parser = new RecSys19DataParser();
            parser.parseItemData(dataPath + "item_metadata.csv");
            parser.parseSessionData(dataPath + "train.csv", dataPath + "test.csv");
            parser.createSplit(outPath + "valid.parsed");
            MLIOUtils.writeObjectToFile(parser.data, outPath + "data.parsed");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
//train
//RecSys19DataParser: session count 910732 elapsed [1 min 0 sec]
//RecSys19DataParser: cur session index 15932991 elapsed [1 min 0 sec]
//RecSys19DataParser: range 1541030408 - 1541548799 elapsed [1 min 0 sec]
//
//test
//RecSys19DataParser: session count 291381 elapsed [1 min 12 sec]
//RecSys19DataParser: cur session index 19715326 elapsed [1 min 12 sec]
//RecSys19DataParser: range 1541548807 - 1541721599 elapsed [1 min 12 sec]