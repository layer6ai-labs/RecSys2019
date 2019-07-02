package recsys2019;
import common.feature.MLSparseFeature;
import common.feature.MLFeatureTransform;
import common.linalg.MLDenseVector;
import common.linalg.MLSparseMatrix;
import common.linalg.MLSparseVector;
import common.utils.MLTimer;
import recsys2019.RecSys19Data.ItemFeature;
import recsys2019.RecSys19Data.SessionFeature;
import recsys2019.RecSys19Model.RecSys19Config;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

public class RecSys19FeatureExtractor {

    public static class SessionInstance {
        public int itemIndex;
        public int target;
        public MLSparseVector features;
    }

    private static MLTimer timer;

    static {
        timer = new MLTimer("RecSys19FeatureExtractor");
        timer.tic();
    }

    private RecSys19Data data;
    private RecSys19Config config;
    private MLSparseMatrix actionRUser;
    private MLSparseMatrix actionRUserNorm;
    private MLSparseMatrix actionRUserNormT;
    private MLSparseMatrix impressRUser;
    private MLSparseMatrix impressRUserNorm;
    private MLSparseMatrix impressRUserNormT;
    private float[][] itemCounts;
    private float[][] userCounts;
    private float[][] itemPrices;
    private MLSparseMatrix itemProperties;
    private float[][] platformCounts;
    private float[][] cityCounts;
    private float[][] deviceCounts;
    private float[][] rankCounts;
    private float[][] priceRankCounts;
    private MLSparseFeature propFeatMatrix;
    private Map<Integer, String> propIndexToCat;
    private int itemActionsLength = 0;

    public RecSys19FeatureExtractor(final RecSys19Data dataP,
                                    final RecSys19Config configP) throws Exception {
        this.data = dataP;
        this.config = configP;
        this.itemProperties = this.data.itemFeatures.get(ItemFeature.properties).getFeatMatrix();
        MLFeatureTransform colSelector = new MLFeatureTransform.ColSelectorTransform(1_000);
        colSelector.apply(this.itemProperties);
        this.propFeatMatrix = this.data.itemFeatures.get(ItemFeature.properties);
        this.propIndexToCat = this.propFeatMatrix.getIndexToCat();
        this.initMatrices();
        this.initCache();
    }

    private void initCache() {
        String[] itemActions = new String[]{
                RecSys19Data.CLICKOUT_ITEM_ACTION,
                RecSys19Data.INTERACTION_ITEM_RATING_ACTION,
                RecSys19Data.INTERACTION_ITEM_INFO_ACTION,
                RecSys19Data.INTERACTION_ITEM_IMAGE_ACTION,
                RecSys19Data.INTERACTION_ITEM_DEALS_ACTION,
                RecSys19Data.SEARCH_FOR_ITEM_ACTION
        };
        this.itemActionsLength = itemActions.length;
        Map<Integer, Integer> actionToIndex = new HashMap();
        for (int i = 0; i < itemActions.length; i++) {
            Integer action = RecSys19Helper.getActionIndex(itemActions[i], this.data);
            actionToIndex.put(action, i);
        }
        int nActionsTotal = this.data.sessionFeatures.get(SessionFeature.action_type).getFeatMatrix().getNCols();
        this.itemCounts = new float[this.data.itemIdToIndex.size()][itemActions.length + 15];
        this.userCounts = new float[this.data.userToSessionStart.size()][nActionsTotal + 3];
        this.itemPrices = new float[this.data.itemIdToIndex.size()][3];
        this.platformCounts = new float[this.data.sessionFeatures.get(SessionFeature.platform).getFeatMatrix().getNCols()][2];
        this.cityCounts = new float[this.data.sessionFeatures.get(SessionFeature.city).getFeatMatrix().getNCols()][5];
        this.deviceCounts = new float[this.data.sessionFeatures.get(SessionFeature.device).getFeatMatrix().getNCols()][2];
        this.rankCounts = new float[25][6];
        this.priceRankCounts = new float[25][6];
        IntStream.range(0, this.data.referenceItems.length).parallel().forEach(index -> {
            boolean isSkipIndex = RecSys19Helper.isSkipIndex(index, this.config, this.data);
            int action = RecSys19Helper.getIndex(index, SessionFeature.action_type, this.data);
            int userIndex = RecSys19Helper.getIndex(index, SessionFeature.user_id, this.data);
            int platformIndex = RecSys19Helper.getIndex(index, SessionFeature.platform, this.data);
            int cityIndex = RecSys19Helper.getIndex(index, SessionFeature.city, this.data);
            int deviceIndex = RecSys19Helper.getIndex(index, SessionFeature.device, this.data);
            if (isSkipIndex == false) {
                synchronized (this.userCounts[userIndex]) {
                    this.userCounts[userIndex][action]++;
                }
            }
            Integer actionIndex = actionToIndex.get(action);
            if (actionIndex == null) {
                return;
            }
            int itemIndex = this.data.referenceItems[index];
            if (itemIndex >= 0 && isSkipIndex == false) {
                synchronized (this.itemCounts[itemIndex]) {
                    this.itemCounts[itemIndex][actionIndex]++;
                }
            }
            int[] impressions = this.data.impressions[index];
            if (impressions != null) {
                int[] prices = this.data.prices[index];
                int[] priceRanking = RecSys19Helper.argsort(prices, true, true);
                int[] priceRankingAbove = RecSys19Helper.getPriceRankAbove(prices);
                int[] stars = RecSys19Helper.getStars(impressions, this.propFeatMatrix, this.propIndexToCat);
                int[] ratings = RecSys19Helper.getRatings(impressions, this.propFeatMatrix, this.propIndexToCat);
                float medianPrice = RecSys19Helper.computeMedianPrice(prices, priceRanking);
                float[] ratingCount = new float[5];
                for (int i = 0; i < ratings.length; i++) {
                    ratingCount[ratings[i]]++;
                }
                float[] starCount = new float[6];
                for (int i = 0; i < stars.length; i++) {
                    starCount[stars[i]]++;
                }
                HashMap<Integer, Integer[]> idxStarsMap = RecSys19Helper.getIdxStarMap(impressions, stars);
                HashMap<Integer, Integer[]> idxRatingsMap = RecSys19Helper.getIdxRatingMap(impressions, ratings);
                HashMap<String, float[]> priceStatsStarsMap = RecSys19Helper.getPriceStatsStarsMap(idxStarsMap, prices, priceRanking);
                float[] priceAverageStar = priceStatsStarsMap.get("priceAverageStar");
                float[] priceRankAverageStar = priceStatsStarsMap.get("priceRankAverageStar");
                HashMap<String, float[]> priceStatsRatingsMap = RecSys19Helper.getPriceStatsRatingsMap(idxRatingsMap, prices, priceRanking);
                float[] priceAverageRating = priceStatsRatingsMap.get("priceAverageRating");
                float[] priceRankAverageRating = priceStatsRatingsMap.get("priceRankAverageRating");
                HashMap<String, HashMap<Integer, Integer>> starRankLocalMaps = RecSys19Helper.getStarRankingMaps(idxStarsMap, prices);
                HashMap<Integer, Integer> rankLocalStarsMap = starRankLocalMaps.get("rankLocalStars");
                HashMap<Integer, Integer> priceRankLocalStarsMap = starRankLocalMaps.get("priceRankLocalStars");
                HashMap<String, HashMap<Integer, Integer>> ratingRankLocalMaps = RecSys19Helper.getRatingRankingMaps(idxRatingsMap, prices);
                HashMap<Integer, Integer> rankLocalRatingsMap = ratingRankLocalMaps.get("rankLocalRatings");
                HashMap<Integer, Integer> priceRankLocalRatingsMap = ratingRankLocalMaps.get("priceRankLocalRatings");
                for (int i = 0; i < impressions.length; i++) {
                    synchronized (this.itemCounts[impressions[i]]) {
                        this.itemCounts[impressions[i]][itemActions.length]++;
                        this.itemCounts[impressions[i]][itemActions.length + 1] += (1.0f + i);
                        this.itemCounts[impressions[i]][itemActions.length + 2] += priceRanking[i];
                        this.itemCounts[impressions[i]][itemActions.length + 3] += priceRankingAbove[i];
                        this.itemCounts[impressions[i]][itemActions.length + 4] += rankLocalStarsMap.get(i) / starCount[stars[i]];
                        this.itemCounts[impressions[i]][itemActions.length + 5] += priceRankLocalStarsMap.get(i) / starCount[stars[i]];
                        this.itemCounts[impressions[i]][itemActions.length + 6] += rankLocalRatingsMap.get(i) / ratingCount[ratings[i]];
                        this.itemCounts[impressions[i]][itemActions.length + 7] += priceRankLocalRatingsMap.get(i) / ratingCount[ratings[i]];
                        this.itemCounts[impressions[i]][itemActions.length + 8] += priceRanking[i] - priceRankAverageStar[stars[i]];
                        this.itemCounts[impressions[i]][itemActions.length + 9] += priceRankAverageStar[stars[i]];
                        this.itemCounts[impressions[i]][itemActions.length + 10] += priceRanking[i] - priceRankAverageRating[ratings[i]];
                        this.itemCounts[impressions[i]][itemActions.length + 11] += priceRankAverageRating[ratings[i]];
                        this.itemCounts[impressions[i]][itemActions.length + 12] += prices[i] - medianPrice;
                        this.itemCounts[impressions[i]][itemActions.length + 13] += medianPrice;
                    }
                    synchronized (this.rankCounts[i]) {
                        this.rankCounts[i][0]++;
                        this.rankCounts[i][1] += priceRanking[i];
                        this.rankCounts[i][2] += prices[i] - priceAverageStar[stars[i]];
                        this.rankCounts[i][3] += prices[i] - priceAverageRating[ratings[i]];
                        this.rankCounts[i][4] += stars[i];
                        this.rankCounts[i][5] += ratings[i];
                    }
                    synchronized (this.priceRankCounts[priceRanking[i]-1]) {
                        this.priceRankCounts[priceRanking[i]-1][0]++;
                        this.priceRankCounts[priceRanking[i]-1][1] += (1.0f + i);
                        this.priceRankCounts[priceRanking[i]-1][2] += prices[i] - priceAverageStar[stars[i]];
                        this.priceRankCounts[priceRanking[i]-1][3] += prices[i] - priceAverageRating[ratings[i]];
                        this.priceRankCounts[priceRanking[i]-1][4] += stars[i];
                        this.priceRankCounts[priceRanking[i]-1][5] += ratings[i];
                    }
                    synchronized (this.platformCounts[platformIndex]) {
                        if (i < 5) {
                            this.platformCounts[platformIndex][0]++;
                            this.platformCounts[platformIndex][1] += priceRanking[i];
                        }
                    }
                    synchronized (this.cityCounts[cityIndex]) {
                        if (i < 5) {
                            this.cityCounts[cityIndex][0]++;
                            this.cityCounts[cityIndex][1] += priceRanking[i];
                            this.cityCounts[cityIndex][2] += prices[i];
                            this.cityCounts[cityIndex][3] += stars[i];
                            this.cityCounts[cityIndex][4] += ratings[i];
                        }
                    }
                    synchronized (this.deviceCounts[deviceIndex]) {
                        if (i < 5) {
                            this.deviceCounts[deviceIndex][0]++;
                            this.deviceCounts[deviceIndex][1] += priceRanking[i];
                        }
                    }
                    synchronized (this.itemPrices[impressions[i]]) {
                        this.itemPrices[impressions[i]][0] += prices[i];
                        if (this.itemPrices[impressions[i]][1] == 0.0f || prices[i] > this.itemPrices[impressions[i]][1]) {
                            this.itemPrices[impressions[i]][1] = prices[i];
                        }
                        if (this.itemPrices[impressions[i]][2] == 0.0f || prices[i] < this.itemPrices[impressions[i]][2]) {
                            this.itemPrices[impressions[i]][2] = prices[i];
                        }
                    }
                    if (itemIndex == impressions[i] && isSkipIndex == false) {
                        synchronized (this.userCounts[userIndex]) {
                            this.userCounts[userIndex][nActionsTotal]++;
                            this.userCounts[userIndex][nActionsTotal + 1] += (1.0 + i);
                            this.userCounts[userIndex][nActionsTotal + 2] += priceRanking[i];
                        }
                    }
                }
            }
        });
        for (int i = 0; i < this.itemCounts.length; i++) {
            MLSparseVector row = this.actionRUserNormT.getRow(i, false);
            if (row != null) {
                this.itemCounts[i][this.itemCounts[i].length - 1] = row.getIndexes().length;
            }
            float count = this.itemCounts[i][itemActions.length];
            if (count > 1) {
                for (int j = 1; j <= 13; j++) {
                    this.itemCounts[i][itemActions.length + j] /= count;
                }
            }
        }
        for (int i = 0; i < this.userCounts.length; i++) {
            float count = this.userCounts[i][nActionsTotal];
            if (count > 1) {
                this.userCounts[i][nActionsTotal + 1] /= count;
                this.userCounts[i][nActionsTotal + 2] /= count;
            }
        }
        for (int i = 0; i < this.rankCounts.length; i++) {
            float count = this.rankCounts[i][0];
            if (count > 1) {
                for (int j = 1; j < this.rankCounts[i].length; j++) {
                    this.rankCounts[i][j] /= count;
                }
            }
        }
        for (int i = 0; i < this.priceRankCounts.length; i++) {
            float count = this.priceRankCounts[i][0];
            if (count > 1) {
                for (int j = 1; j < this.priceRankCounts[i].length; j++) {
                    this.priceRankCounts[i][j] /= count;
                }
            }
        }
        for (int i = 0; i < this.platformCounts.length; i++) {
            float count = this.platformCounts[i][0];
            if (count > 1) {
                for (int j = 1; j < this.platformCounts[i].length; j++) {
                    this.platformCounts[i][j] /= count;
                }
            }
        }
        for (int i = 0; i < this.cityCounts.length; i++) {
            float count = this.cityCounts[i][0];
            if (count > 1) {
                for (int j = 1; j < this.cityCounts[i].length; j++) {
                    this.cityCounts[i][j] /= count;
                }
            }
        }
        for (int i = 0; i < this.deviceCounts.length; i++) {
            float count = this.deviceCounts[i][0];
            if (count > 1) {
                for (int j = 1; j < this.deviceCounts[i].length; j++) {
                    this.deviceCounts[i][j] /= count;
                }
            }
        }
        for (int i = 0; i < this.itemPrices.length; i++) {
            float count = this.itemCounts[i][itemActions.length + 1];
            if (count > 1) {
                this.itemPrices[i][0] /= count;
            }
        }
        timer.toc("initCache done");
    }

    private void initMatrices() {
        this.actionRUser = RecSys19Helper.createUserMatrix(new String[]{
                        RecSys19Data.CLICKOUT_ITEM_ACTION,
                        RecSys19Data.INTERACTION_ITEM_RATING_ACTION,
                        RecSys19Data.INTERACTION_ITEM_INFO_ACTION,
                        RecSys19Data.INTERACTION_ITEM_IMAGE_ACTION,
                        RecSys19Data.INTERACTION_ITEM_DEALS_ACTION,
                        RecSys19Data.SEARCH_FOR_ITEM_ACTION}, this.data, this.config);
        this.actionRUser.binarizeValues();
        this.actionRUserNorm = this.actionRUser.deepCopy();
        actionRUserNorm.applyRowNorm(actionRUserNorm.getRowNorm(2));
        actionRUserNorm.applyColNorm(actionRUserNorm.getColNorm(2));
        this.actionRUserNormT = this.actionRUserNorm.transpose();
        this.impressRUser = RecSys19Helper.createUserMatrix(null, this.data, this.config);
        this.impressRUser.binarizeValues();
        this.impressRUserNorm = this.impressRUser.deepCopy();
        impressRUserNorm.applyRowNorm(impressRUserNorm.getRowNorm(2));
        impressRUserNorm.applyColNorm(impressRUserNorm.getColNorm(2));
        this.impressRUserNormT = this.impressRUserNorm.transpose();
        timer.toc("initMatrices done");
    }

    public List<MLSparseVector> getItemFeatures(final int itemIndex) {
        List<MLSparseVector> feats = new LinkedList();
        feats.add(new MLDenseVector(this.itemCounts[itemIndex]).toSparse());
        feats.add(this.itemProperties.getRow(itemIndex, true));
       return feats;
    }

    public List<MLSparseVector> getUserFeatures(final int userIndex) {
        List<MLSparseVector> feats = new LinkedList();
        float[] userCount = this.userCounts[userIndex].clone();
        feats.add(new MLDenseVector(userCount).toSparse());
        return feats;
    }

    public List<MLSparseVector> getUserItemFeatures(final int targetUserIndex,
                                                    final int targetItemIndex) {
        List<MLSparseVector> feats = new LinkedList();
        return feats;
    }

    public List<MLSparseVector> getSessionFeatures(final int targetUserIndex,
                                                   final int targetItemIndex,
                                                   final int targetSessionStart,
                                                   final int targetSessionEnd) {
        List<MLSparseVector> feats = new LinkedList();
        int nActionsTotal = this.data.sessionFeatures.get(SessionFeature.action_type).getFeatMatrix().getNCols();
        float[][] actionFeats = new float[nActionsTotal][];
        String[] actions = new String[]{
                RecSys19Data.CLICKOUT_ITEM_ACTION,
                RecSys19Data.INTERACTION_ITEM_RATING_ACTION,
                RecSys19Data.INTERACTION_ITEM_INFO_ACTION,
                RecSys19Data.INTERACTION_ITEM_IMAGE_ACTION,
                RecSys19Data.INTERACTION_ITEM_DEALS_ACTION,
                RecSys19Data.SEARCH_FOR_ITEM_ACTION};
        for (String itemAction : actions) {
            int actionIndex = RecSys19Helper.getActionIndex(itemAction, this.data);
            actionFeats[actionIndex] = new float[2];
        }
        int[] targetImpressions = this.data.impressions[targetSessionEnd];
        int[] targetPrices = this.data.prices[targetSessionEnd];
        int[] targetPriceRank = RecSys19Helper.argsort(targetPrices, true, true);
        int[] targetPriceRankAbove = RecSys19Helper.getPriceRankAbove(targetPrices);
        int[] targetStars = RecSys19Helper.getStars(targetImpressions, this.propFeatMatrix, this.propIndexToCat);
        int[] targetRatings = RecSys19Helper.getRatings(targetImpressions, this.propFeatMatrix, this.propIndexToCat);
        int[] targetStarRank = RecSys19Helper.argsort(targetStars, false, true);
        int[] targetRatingRank = RecSys19Helper.argsort(targetRatings, false, true);
        int targetItemRank = 0;
        int targetItemPrice = 0;
        int targetItemPriceRank = 0;
        int targetItemPriceRankAbove = 0;
        for (int i = 0; i < targetImpressions.length; i++) {
            if (targetImpressions[i] == targetItemIndex) {
                targetItemRank = 1 + i;
                targetItemPrice = targetPrices[i];
                targetItemPriceRank = targetPriceRank[i];
                targetItemPriceRankAbove = targetPriceRankAbove[i];
                break;
            }
        }
        feats.add(new MLDenseVector(new float[]{
                this.itemPrices[targetItemIndex][0] - targetItemPrice,
                this.itemPrices[targetItemIndex][1] - targetItemPrice,
                this.itemPrices[targetItemIndex][2] - targetItemPrice
        }).toSparse());
        feats.add(new MLDenseVector(new float[]{
                this.itemCounts[targetItemIndex][this.itemActionsLength + 1] - targetItemRank,
                this.itemCounts[targetItemIndex][this.itemActionsLength + 2] - targetItemPriceRank,
                this.itemCounts[targetItemIndex][this.itemActionsLength + 3] - targetItemPriceRankAbove,
        }).toSparse());
        float[] sameImpress = new float[2];
        for (int index = targetSessionStart; index < targetSessionEnd; index++) {
            int itemIndex = this.data.referenceItems[index];
            if (itemIndex >= 0) {
                if (RecSys19Helper.sameImpressions(targetImpressions, this.data.impressions[index], true) == true) {
                    sameImpress[0]++;
                    if (this.data.referenceItems[index] == targetItemIndex) {
                        sameImpress[1]++;
                    }
                }
            }
        }
        feats.add(new MLDenseVector(sameImpress).toSparse());
        feats.add(new MLDenseVector(new float[]{
                this.userCounts[targetUserIndex][nActionsTotal + 1] - targetItemRank,
                this.userCounts[targetUserIndex][nActionsTotal + 2] - targetItemPriceRank,
        }).toSparse());
        final int MAX_ITEM_LAG = 2;
        float[][] lastItemSim = new float[MAX_ITEM_LAG][8];
        int curItemLag = 0;
        final int MAX_ACTION_LAG = 1;
        float[][] lastActionSim = new float[MAX_ACTION_LAG][2];
        int curActionLag = 0;
        for (int index = targetSessionEnd - 1; index >= targetSessionStart; index--) {
            int itemIndex = this.data.referenceItems[index];
            if (itemIndex < 0) {
                int action = RecSys19Helper.getIndex(index, SessionFeature.action_type, this.data);
                if (curActionLag < MAX_ACTION_LAG) {
                    lastActionSim[curActionLag][0] = action;
                    lastActionSim[curActionLag][1] = this.data.timeStamps[index + 1] - this.data.timeStamps[index];
                    curActionLag++;
                }
                continue;
            }
            int rank = 0;
            int priceRank = 0;
            for (int i = 0; i < targetImpressions.length; i++) {
                if (itemIndex == targetImpressions[i]) {
                    rank = i + 1;
                    priceRank = targetPriceRank[i];
                    break;
                }
            }
            if (curItemLag < MAX_ITEM_LAG) {
                float[] stats = new float[2];
                for (int i = index - 1; i >= targetSessionStart; i--) {
                    if (this.data.referenceItems[i] == itemIndex) {
                        stats[0]++;
                        stats[1] += (this.data.timeStamps[i + 1] - this.data.timeStamps[i]);
                    }
                }
                int action = RecSys19Helper.getIndex(index, SessionFeature.action_type, this.data);
                int cur = 0;
                lastItemSim[curItemLag][cur] = priceRank;
                cur++;
                lastItemSim[curItemLag][cur] = targetItemPriceRank - priceRank;
                cur++;
                lastItemSim[curItemLag][cur] = rank;
                cur++;
                lastItemSim[curItemLag][cur] = targetItemRank - rank;
                cur++;
                lastItemSim[curItemLag][cur] = action;
                cur++;
                lastItemSim[curItemLag][cur] = targetSessionEnd - index;
                cur++;
                lastItemSim[curItemLag][cur] = this.data.timeStamps[index + 1] - this.data.timeStamps[index];
                cur++;
                lastItemSim[curItemLag][cur] = stats[1];
                curItemLag++;
            }
        }
        for (int i = 0; i < lastItemSim.length; i++) {
            feats.add(new MLDenseVector(lastItemSim[i]).toSparse());
        }
        for (int i = 0; i < lastActionSim.length; i++) {
            feats.add(new MLDenseVector(lastActionSim[i]).toSparse());
        }
        return feats;
    }

    public float[] getColdWarmSessionStats(final int targetUserIndex,
                                           final int targetItemIndex,
                                           final int targetSessionStart,
                                           final int targetSessionEnd) {
        int[] targetImpressions = this.data.impressions[targetSessionEnd];
        int[] targetPrices = this.data.prices[targetSessionEnd];
        int[] targetPriceRank = RecSys19Helper.argsort(targetPrices, true, true);
        int targetItemRank = 0;
        int targetItemPriceRank = 0;
        for (int i = 0; i < targetImpressions.length; i++) {
            if (targetImpressions[i] == targetItemIndex) {
                targetItemRank = 1 + i;
                targetItemPriceRank = targetPriceRank[i];
                break;
            }
        }
        int actionItemClickoutIndex = RecSys19Helper.getActionIndex(RecSys19Data.CLICKOUT_ITEM_ACTION, this.data);
        int nClickouts = 0;
        int nSteps = 0;
        for (int index = targetSessionStart; index <= targetSessionEnd; index++) {
            int action = RecSys19Helper.getIndex(index, SessionFeature.action_type, this.data);
            if (action == actionItemClickoutIndex) {
                nClickouts++;
            }
            nSteps++;
        }
        float[] topRankSignals = new float[5];
        if (targetItemRank <= topRankSignals.length && targetItemRank >= 1) {
            topRankSignals[targetItemRank - 1] = 1;
        }
        float[] topPriceRankSignals = new float[5];
        if (targetItemPriceRank <= topPriceRankSignals.length && targetItemPriceRank >= 1) {
            topPriceRankSignals[targetItemPriceRank - 1] = 1;
        }
        HashMap<Integer, Integer> impressionToRankMap = new HashMap<>();
        for (int i = 0; i < targetImpressions.length; i++) {
            impressionToRankMap.put(targetImpressions[i], i+1);
        }
        String[] actionNames = new String[] {
                RecSys19Data.CLICKOUT_ITEM_ACTION,
                RecSys19Data.INTERACTION_ITEM_RATING_ACTION,
                RecSys19Data.INTERACTION_ITEM_INFO_ACTION,
                RecSys19Data.INTERACTION_ITEM_IMAGE_ACTION,
                RecSys19Data.INTERACTION_ITEM_DEALS_ACTION,
                RecSys19Data.SEARCH_FOR_ITEM_ACTION,
        };
        HashMap<Integer, Integer> actionToIndexMap = new HashMap<>();
        for (int i = 0; i < actionNames.length; i++) {
            actionToIndexMap.put(RecSys19Helper.getActionIndex(actionNames[i], this.data), i);
        }
        int[] nInteractAboveCounts = new int[2];
        int[] nInteractEqualCounts = new int[2];
        int[] nInteractBelowCounts = new int[2];
        for (int index = targetSessionStart; index <= targetSessionEnd - 1; index++) {
            int referenceItemIndex = this.data.referenceItems[index];
            int action = RecSys19Helper.getIndex(index, SessionFeature.action_type, this.data);
            Integer actionIndex = actionToIndexMap.get(action);
            if (actionIndex == null) {
                continue;
            }
            int actionIndexMacro = actionIndex == 0 ? 0:1;
            if (referenceItemIndex > 0) {
                Integer referenceRank = impressionToRankMap.get(referenceItemIndex);
                if (referenceRank == null) {
                    continue;
                }
                if (referenceRank < targetItemRank) {
                    nInteractAboveCounts[actionIndexMacro]++;
                }
                if (referenceRank == targetItemRank) {
                    nInteractEqualCounts[actionIndexMacro]++;
                }
                if (referenceRank > targetItemRank) {
                    nInteractBelowCounts[actionIndexMacro]++;
                }
            }
        }
        float[] nInteractAboveCountsFloat = RecSys19Helper.convertToFloats(nInteractAboveCounts);
        float[] nInteractEqualCountsFloat = RecSys19Helper.convertToFloats(nInteractEqualCounts);
        float[] nInteractBelowCountsFloat = RecSys19Helper.convertToFloats(nInteractBelowCounts);
        return RecSys19Helper.combine(
                topRankSignals,
                topPriceRankSignals,
                nInteractAboveCountsFloat,
                nInteractEqualCountsFloat,
                nInteractBelowCountsFloat
        );
    }

    public SessionInstance[] extractFeatures(final int targetIndex) {
        int[] impressions = this.data.impressions[targetIndex];
        int[] prices = this.data.prices[targetIndex];
        int[] priceRanking = RecSys19Helper.argsort(prices, true, true);
        int[] priceRankingAbove = RecSys19Helper.getPriceRankAbove(prices);
        int[] stars = RecSys19Helper.getStars(impressions, this.propFeatMatrix, this.propIndexToCat);
        int[] ratings = RecSys19Helper.getRatings(impressions, this.propFeatMatrix, this.propIndexToCat);
        float[] ratingCount = new float[5];
        for (int i = 0; i < ratings.length; i++) {
            ratingCount[ratings[i]]++;
        }
        float[] starCount = new float[6];
        for (int i = 0; i < stars.length; i++) {
            starCount[stars[i]]++;
        }
        int userIndex = RecSys19Helper.getIndex(targetIndex, SessionFeature.user_id, this.data);
        int platformIndex = RecSys19Helper.getIndex(targetIndex, SessionFeature.platform, this.data);
        int cityIndex = RecSys19Helper.getIndex(targetIndex, SessionFeature.city, this.data);
        int deviceIndex = RecSys19Helper.getIndex(targetIndex, SessionFeature.device, this.data);
        int targetItem = this.data.referenceItems[targetIndex];
        int step = (int) RecSys19Helper.getValue(targetIndex, SessionFeature.step, this.data);
        int sessionStart = RecSys19Helper.getSessionStartIndex(targetIndex, this.data);
        float[] uuUserAction = RecSys19Helper.getUserUser(userIndex, impressions, this.actionRUserNorm, this.actionRUserNormT);
        float[] iiUserAction = RecSys19Helper.getItemItem(userIndex, impressions, this.actionRUser, this.actionRUserNormT);
        float[] uuUserImpress = RecSys19Helper.getUserUser(userIndex, impressions, this.impressRUserNorm, this.impressRUserNormT);
        float[] iiUserImpress = RecSys19Helper.getItemItem(userIndex, impressions, this.impressRUser, this.impressRUserNormT);
        float[] meanScores = new float[6];
        for (int i = 0; i < uuUserImpress.length; i++) {
            meanScores[0] += uuUserAction[i];
            meanScores[1] += iiUserAction[i];
            meanScores[2] += uuUserImpress[i];
            meanScores[3] += iiUserImpress[i];
            if (i > 0) {
                meanScores[4] += RecSys19Helper.getItemItemForItem(impressions[i], impressions[i - 1], this.actionRUserNormT);
                meanScores[5] += RecSys19Helper.getItemItemForItem(impressions[i], impressions[i - 1], this.impressRUserNormT);
            }
        }
        float[] scoreEntropy = new float[4];
        for (int i = 0; i < impressions.length; i++) {
            float rel = uuUserAction[i] / (meanScores[0] == 0 ? 1 : meanScores[0]);
            if (rel > 0) {
                scoreEntropy[0] += rel * Math.log(rel);
            }
            rel = iiUserAction[i] / (meanScores[1] == 0 ? 1 : meanScores[1]);
            if (rel > 0) {
                scoreEntropy[1] += rel * Math.log(rel);
            }
            rel = uuUserImpress[i] / (meanScores[2] == 0 ? 1 : meanScores[2]);
            if (rel > 0) {
                scoreEntropy[2] += rel * Math.log(rel);
            }
            rel = iiUserImpress[i] / (meanScores[3] == 0 ? 1 : meanScores[3]);
            if (rel > 0) {
                scoreEntropy[3] += rel * Math.log(rel);
            }
        }
        float[] meanGlobal = new float[this.itemCounts[0].length];
        for (int i = 0; i < impressions.length; i++) {
            float[] itemGlobal = this.itemCounts[impressions[i]];
            for (int j = 0; j < itemGlobal.length; j++) {
                meanGlobal[j] += itemGlobal[j];
            }
        }
        HashMap<Integer, Integer[]> idxStarsMap = RecSys19Helper.getIdxStarMap(impressions, stars);
        HashMap<Integer, Integer[]> idxRatingsMap = RecSys19Helper.getIdxRatingMap(impressions, ratings);
        HashMap<String, float[]> priceStatsStarsMap = RecSys19Helper.getPriceStatsStarsMap(idxStarsMap, prices, priceRanking);
        float[] priceAverageStar = priceStatsStarsMap.get("priceAverageStar");
        float[] priceRankAverageStar = priceStatsStarsMap.get("priceRankAverageStar");
        HashMap<String, float[]> priceStatsRatingsMap = RecSys19Helper.getPriceStatsRatingsMap(idxRatingsMap, prices, priceRanking);
        float[] priceAverageRating = priceStatsRatingsMap.get("priceAverageRating");
        float[] priceRankAverageRating = priceStatsRatingsMap.get("priceRankAverageRating");
        HashMap<String, HashMap<Integer, Integer>> starRankLocalMaps = RecSys19Helper.getStarRankingMaps(idxStarsMap, prices);
        HashMap<Integer, Integer> rankLocalStarsMap = starRankLocalMaps.get("rankLocalStars");
        HashMap<Integer, Integer> priceRankLocalStarsMap = starRankLocalMaps.get("priceRankLocalStars");
        HashMap<String, HashMap<Integer, Integer>> ratingRankLocalMaps = RecSys19Helper.getRatingRankingMaps(idxRatingsMap, prices);
        HashMap<Integer, Integer> rankLocalRatingsMap = ratingRankLocalMaps.get("rankLocalRatings");
        HashMap<Integer, Integer> priceRankLocalRatingsMap = ratingRankLocalMaps.get("priceRankLocalRatings");
        float[] meanTopPrices = RecSys19Helper.getTopKMeans(prices, new int[]{
                1, 2, 3, 5, 10, 15, 20, 25
        });
        float[] meanTopPriceRanks = RecSys19Helper.getTopKMeans(priceRanking, new int[]{
                3, 5, 10, 15, 20, 25
        });
        float medianPrice = RecSys19Helper.computeMedianPrice(prices, priceRanking);
        float[] meanProperties = new float[this.itemProperties.getNCols()];
        for (int i = 0; i < impressions.length; i++) {
            int[] indexes = this.itemProperties.getRow(impressions[i], true).getIndexes();
            float[] values = this.itemProperties.getRow(impressions[i], true).getValues();
            if (indexes != null) {
                for (int j = 0; j < indexes.length; j++) {
                    meanProperties[indexes[j]] += values[j];
                }
            }
        }
        float propertyEntropy = 0;
        for (int i = 0; i < impressions.length; i++) {
            int[] indexes = this.itemProperties.getRow(impressions[i], true).getIndexes();
            float[] values = this.itemProperties.getRow(impressions[i], true).getValues();
            if (indexes != null) {
                for (int j = 0; j < indexes.length; j++) {
                    float rel = values[j] / (meanProperties[indexes[j]] == 0 ? 1 : meanProperties[indexes[j]]);
                    if (rel > 0) {
                        propertyEntropy += rel * Math.log(rel);
                    }
                }
            }
        }
        SessionInstance[] instances = new SessionInstance[impressions.length];
        for (int i = 0; i < impressions.length; i++) {
            int itemIndex = impressions[i];
            SessionInstance instance = new SessionInstance();
            instance.itemIndex = itemIndex;
            instance.target = 0;
            if (targetItem == itemIndex) {
                instance.target = 1;
            }
            List<MLSparseVector> features = new LinkedList();
            features.addAll(this.getUserFeatures(userIndex));
            features.addAll(this.getItemFeatures(itemIndex));
            features.addAll(this.getUserItemFeatures(userIndex, itemIndex));
            features.addAll(this.getSessionFeatures(userIndex, itemIndex, sessionStart, targetIndex));
            features.add(new MLDenseVector(new float[]{
                    i + 1.0f,
                    prices[i],
                    priceRanking[i],
                    impressions.length,
                    step,
                    this.data.timeStamps[targetIndex] - this.data.timeStamps[sessionStart],
                    uuUserAction[i],
                    iiUserAction[i],
                    uuUserImpress[i],
                    iiUserImpress[i],
                    propertyEntropy,
                    priceRankingAbove[i],
                    prices[i] - medianPrice,
            }).toSparse());
            features.add(new MLDenseVector(meanScores).toSparse());
            features.add(new MLDenseVector(meanGlobal).toSparse());
            features.add(new MLDenseVector(scoreEntropy).toSparse());
            features.add(new MLDenseVector(this.getColdWarmSessionStats(userIndex, itemIndex, sessionStart, targetIndex)).toSparse());
            features.add(new MLDenseVector(new float[]{
                    rankLocalStarsMap.get(i),
                    priceRankLocalStarsMap.get(i),
                    starCount[stars[i]],
                    rankLocalRatingsMap.get(i),
                    priceRankLocalRatingsMap.get(i),
                    ratingCount[ratings[i]],
            }).toSparse());
            features.add(new MLDenseVector(meanTopPrices).toSparse());
            features.add(new MLDenseVector(meanTopPriceRanks).toSparse());
            features.add(new MLDenseVector(new float[]{
                    priceAverageStar[stars[i]],
                    priceRankAverageStar[stars[i]],
                    priceAverageRating[ratings[i]],
                    priceRankAverageRating[ratings[i]],
                    (prices[i] - priceAverageStar[stars[i]]) / priceAverageStar[stars[i]],
                    priceRanking[i] - priceRankAverageStar[stars[i]],
                    (prices[i] - priceAverageRating[ratings[i]]) / priceAverageRating[ratings[i]],
                    priceRanking[i] - priceRankAverageRating[ratings[i]],
            }).toSparse());
            features.add(new MLDenseVector(this.rankCounts[i]).toSparse());
            features.add(new MLDenseVector(this.priceRankCounts[priceRanking[i]-1]).toSparse());
            features.add(new MLDenseVector(this.platformCounts[platformIndex]).toSparse());
            features.add(new MLDenseVector(this.cityCounts[cityIndex]).toSparse());
            features.add(new MLDenseVector(this.deviceCounts[deviceIndex]).toSparse());
//            features.add(new MLDenseVector(meanProperties).toSparse());
            features.add(this.data.sessionFeatures.get(SessionFeature.device).getRow(targetIndex, true));
            MLSparseVector[] featArr = new MLSparseVector[features.size()];
            featArr = features.toArray(featArr);
            instance.features = MLSparseVector.concat(featArr);
            instances[i] = instance;
        }
        return instances;
    }
}