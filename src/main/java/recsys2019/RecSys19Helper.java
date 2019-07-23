package recsys2019;
import common.feature.MLSparseFeature;
import common.linalg.FloatElement;
import common.linalg.MLDenseVector;
import common.linalg.MLSparseMatrix;
import common.linalg.MLSparseMatrixAOO;
import common.linalg.MLSparseVector;
import common.utils.MLTimer;
import com.google.common.primitives.Ints;
import recsys2019.RecSys19Data.SessionFeature;
import recsys2019.RecSys19Model.RecSys19Config;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class RecSys19Helper {

    private static MLTimer timer;

    static {
        timer = new MLTimer("RecSys19Helper");
        timer.tic();
    }

    public static int getActionIndex(final String action, final RecSys19Data data) {
        return data.sessionFeatures.get(SessionFeature.action_type).getCatToIndex().get(action);
    }

    public static boolean isTrainIndex(final int index, final RecSys19Data data) {
        return Arrays.binarySearch(data.trainEventIndexes, index) >= 0;
    }

    public static boolean isValidIndex(final int index, final RecSys19Data data) {
        return Arrays.binarySearch(data.validEventIndexes, index) >= 0;
    }

    public static boolean isSkipIndex(final int index, final RecSys19Config config, final RecSys19Data data) {
        if (config.removeTrain == true && isTrainIndex(index, data) == true) {
            return true;
        }
        if (config.removeValid == true && isValidIndex(index, data) == true) {
            return true;
        }
        return false;
    }

    public static int getIndex(final int rowIndex, final SessionFeature feature, final RecSys19Data data) {
        return data.sessionFeatures.get(feature).getRow(rowIndex, false).getIndexes()[0];
    }

    public static float getValue(final int rowIndex, final SessionFeature feature, final RecSys19Data data) {
        return data.sessionFeatures.get(feature).getRow(rowIndex, false).getValues()[0];
    }

    public static int getSessionStartIndex(final int index, final RecSys19Data data) {
        int step = (int) RecSys19Helper.getValue(index, SessionFeature.step, data);
        int sessionStart = index - step + 1;
        if (RecSys19Helper.getValue(sessionStart, SessionFeature.step, data) != 1) {
            throw new IllegalStateException("step != 1");
        }
        return sessionStart;
    }

    public static MLSparseMatrixAOO createUserMatrix(final String[] actions, final RecSys19Data data, final RecSys19Config config) {
        int[] actionIndexes;
        AtomicBoolean checkSkip = new AtomicBoolean(false);
        if (actions != null) {
            actionIndexes = new int[actions.length];
            for (int i = 0; i < actions.length; i++) {
                actionIndexes[i] = getActionIndex(actions[i], data);
                if (actions[i].equals(RecSys19Data.CLICKOUT_ITEM_ACTION) == true) {
                    checkSkip.set(true);
                }
            }
            Arrays.sort(actionIndexes);
        } else {
            actionIndexes = new int[]{getActionIndex(RecSys19Data.CLICKOUT_ITEM_ACTION, data)};
        }
        final int nUsers = data.sessionFeatures.get(SessionFeature.user_id).getCatToIndex().size();
        final int nItems = data.itemIdToIndex.size();
        MLSparseVector[] rows = new MLSparseVector[nUsers];
        AtomicInteger count = new AtomicInteger(0);
        IntStream.range(0, nUsers).parallel().forEach(userIndex -> {
            int curCount = count.incrementAndGet();
            if (curCount % 500_000 == 0) {
                timer.tocLoop("createUserMatrix", curCount);
            }
            Map<Integer, FloatElement> userItemMap = new TreeMap();
            Set<Integer> userSessions = data.userToSessionStart.get(userIndex);
            for (int sessionStart : userSessions) {
                final int sessionId = getIndex(sessionStart, SessionFeature.session_id, data);
                int curIndex = sessionStart - 1;
                while (true) {
                    curIndex++;
                    if (curIndex >= data.referenceItems.length) {
                        break;
                    }
                    int curSessionId = getIndex(curIndex, SessionFeature.session_id, data);
                    if (sessionId != curSessionId) {
                        break;
                    }
                    if (checkSkip.get() == true && RecSys19Helper.isSkipIndex(curIndex, config, data) == true) {
                        continue;
                    }
                    int curAction = getIndex(curIndex, SessionFeature.action_type, data);
                    if (Arrays.binarySearch(actionIndexes, curAction) < 0) {
                        continue;
                    }
                    if (actions != null) {
                        int curItemId = data.referenceItems[curIndex];
                        if (curItemId < 0) {
                            continue;
                        }
                        FloatElement element = userItemMap.get(curItemId);
                        if (element == null) {
                            element = new FloatElement(curItemId, 1.0f);
                            userItemMap.put(curItemId, element);
                        } else {
                            element.setValue(element.getValue() + 1.0f);
                        }
                    } else {
                        int[] impressions = data.impressions[curIndex];
                        if (impressions == null) {
                            continue;
                        }
                        for (int itemId : impressions) {
                            FloatElement element = userItemMap.get(itemId);
                            if (element == null) {
                                element = new FloatElement(itemId, 1.0f);
                                userItemMap.put(itemId, element);
                            } else {
                                element.setValue(element.getValue() + 1.0f);
                            }
                        }
                    }
                }
            }
            if (userItemMap.size() == 0) {
                return;
            }
            int[] indexes = new int[userItemMap.size()];
            float[] values = new float[userItemMap.size()];
            int cur = 0;
            for (Map.Entry<Integer, FloatElement> entry : userItemMap.entrySet()) {
                indexes[cur] = entry.getValue().getIndex();
                values[cur] = entry.getValue().getValue();
                cur++;
            }
            rows[userIndex] = new MLSparseVector(indexes, values, null, nItems);
        });
        MLSparseMatrixAOO matrix = new MLSparseMatrixAOO(rows, nItems);
        timer.toc("createUserMatrix nnz " + matrix.getNNZ());
        return matrix;
    }

    public static float[] getItemItem(final int targetIndex, final int[] items, final MLSparseMatrix R, final MLSparseMatrix Rt) {
        MLSparseVector targetRow = R.getRow(targetIndex, false);
        if (targetRow == null) {
            return new float[items.length];
        }
        MLDenseVector colAvg = getRowAvg(Rt, targetRow.getIndexes(), false);
        float[] scores = new float[items.length];
        for (int i = 0; i < items.length; i++) {
            scores[i] = colAvg.mult(Rt.getRow(items[i], true));
        }
        return scores;
    }

    public static float[] getUserUser(final int targetIndex, final int[] items, final MLSparseMatrix R, final MLSparseMatrix Rt) {
        MLSparseVector targetRow = R.getRow(targetIndex, false);
        if (targetRow == null) {
            return new float[items.length];
        }
        Set<Integer> intersect = new HashSet();
        for (int itemIndex : targetRow.getIndexes()) {
            MLSparseVector itemRow = Rt.getRow(itemIndex, false);
            if (itemRow == null) {
                continue;
            }
            for (int itemRowIndex : itemRow.getIndexes()) {
                intersect.add(itemRowIndex);
            }
        }
        float[] scores = new float[items.length];
        for (int i = 0; i < items.length; i++) {
            MLSparseVector itemRow = Rt.getRow(items[i], false);
            if (itemRow == null) {
                continue;
            }
            for (int itemRowIndex : itemRow.getIndexes()) {
                if (intersect.contains(itemRowIndex) == false) {
                    continue;
                }
                scores[i] += targetRow.multiply(R.getRow(itemRowIndex, true));
            }
        }
        return scores;
    }

    public static float getItemItemForItem(final int item, final int anotherItem, final MLSparseMatrix Rt) {
        MLSparseVector itemCol = Rt.getRow(item, false);
        if (itemCol == null) {
            return 0f;
        }
        return itemCol.multiply(Rt.getRow(anotherItem, true));
    }

    public static MLDenseVector getRowAvg(final MLSparseMatrix R, final int[] targetIndexes, final boolean normalize) {
        float[] rowAvg = new float[R.getNCols()];
        int count = 0;
        for (int targetIndex : targetIndexes) {
            MLSparseVector row = R.getRow(targetIndex);
            if (row == null) {
                continue;
            }
            count++;
            int[] indexes = row.getIndexes();
            float[] values = row.getValues();
            for (int i = 0; i < indexes.length; i++) {
                rowAvg[indexes[i]] += values[i];
            }
        }
        if (normalize == true && count > 1) {
            for (int i = 0; i < rowAvg.length; i++) {
                rowAvg[i] = rowAvg[i] / count;
            }
        }
        return new MLDenseVector(rowAvg);
    }

    public static int[] argsort(final int[] a, final boolean ascending, final boolean rank) {
        Integer[] indexes = new Integer[a.length];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, new Comparator<Integer>() {
            @Override
            public int compare(final Integer i1, final Integer i2) {
                return (ascending ? 1 : -1) * Ints.compare(a[i1], a[i2]);
            }
        });
        int[] ret = new int[indexes.length];
        for (int i = 0; i < ret.length; i++) {
            if (rank == true) {
                ret[indexes[i]] = i + 1;
            } else {
                ret[i] = indexes[i];
            }
        }
        return ret;
    }

    public static int[] getPriceRankAbove(final int[] prices) {
        int[] priceRankingAbove = new int[prices.length];
        for (int i = 0; i < prices.length; i++) {
            int[] priceSubArray = Arrays.copyOfRange(prices, 0, i + 1);
            priceRankingAbove[i] = RecSys19Helper.argsort(
                    priceSubArray,
                    true,
                    true)[i];
        }
        return priceRankingAbove;
    }

    public static boolean sameImpressions(final int[] impressions1, final int[] impressions2, final boolean inOrder) {
        if (impressions1 == null || impressions2 == null) {
            return false;
        }
        if (impressions1.length != impressions2.length) {
            return false;
        }
        if (inOrder == true) {
            for (int i = 0; i < impressions1.length; i++) {
                if (impressions1[i] != impressions2[i]) {
                    return false;
                }
            }
        } else {
            for (int i = 0; i < impressions1.length; i++) {
                boolean found = false;
                for (int j = 0; j < impressions2.length; j++) {
                    if (impressions1[i] == impressions2[j]) {
                        found = true;
                        break;
                    }
                }
                if (found == false) {
                    return false;
                }
            }
        }
        return true;
    }

    public static int[] getStars(final int[] impressions, final MLSparseFeature propFeatMatrix, final Map<Integer, String> propIndexToCat) {
        int[] stars = new int[impressions.length];
        for (int i = 0; i < impressions.length; i++) {
            MLSparseVector featuresProps = propFeatMatrix.getRow(impressions[i], true);
            int[] propIndexes = featuresProps.getIndexes();
            if (propIndexes == null) {
                continue;
            }
            for (int j = 0; j < propIndexes.length; j++) {
                String prop = propIndexToCat.get(propIndexes[j]);
                if (prop.contains("Star")) {
                    int star = 0;
                    if (prop.equals("1 Star")) { star = 1; }
                    else if (prop.equals("2 Star")) { star = 2; }
                    else if (prop.equals("3 Star")) { star = 3; }
                    else if (prop.equals("4 Star")) { star = 4; }
                    else if (prop.equals("5 Star")) { star = 5; }
                    if (star != 0 && star < stars[i]) {
                        stars[i] = star;
                    }
                }
            }
        }
        return stars;
    }

    public static int[] getRatings(final int[] impressions, final MLSparseFeature propFeatMatrix, final Map<Integer, String> propIndexToCat) {
        int[] ratings = new int[impressions.length];
        for (int i = 0; i < impressions.length; i++) {
            MLSparseVector featuresProps = propFeatMatrix.getRow(impressions[i], true);
            int[] propIndexes = featuresProps.getIndexes();
            if (propIndexes == null) {
                continue;
            }
            for (int j = 0; j < propIndexes.length; j++) {
                String prop = propIndexToCat.get(propIndexes[j]);
                if (prop.contains("Rating")) {
                    int rating = 0;
                    if (prop.equals("Satisfactory Rating")) { rating = 1; }
                    else if (prop.equals("Good Rating")) { rating = 2; }
                    else if (prop.equals("Very Good Rating")) { rating = 3; }
                    else if (prop.equals("Excellent Rating")) { rating = 4; }
                    if (rating != 0 && rating < ratings[i]) {
                        ratings[i] = rating;
                    }
                }
            }
        }
        return ratings;
    }

    public static float[] combine(final float[] ...arrs) {
        int N = 0;
        for (float[] arr : arrs) {
            N += arr.length;
        }
        float[] arrCombined = new float[N];
        int i = 0;
        for (float[] arr : arrs) {
            for (float x : arr) {
                arrCombined[i] = x;
                i++;
            }
        }
        return arrCombined;
    }

    public static int computeMedianPrice(int[] prices, int[] priceRankings) {
        int medianPrice = prices[prices.length - 1];
        int middleRank = prices.length / 2;
        for (int i = 0; i < prices.length; i++) {
            if (priceRankings[i] == middleRank) {
                medianPrice = prices[i];
                break;
            }
        }
        return medianPrice;
    }

    public static float[] convertToFloats(int[] arr) {
        float[] arrFloat = new float[arr.length];
        for (int i = 0 ; i < arr.length; i++) {
            arrFloat[i] = (float) arr[i];
        }
        return arrFloat;
    }

    public static float[] getTopKMeans(int[] arr, int[] k) {
        int[] topKSum = new int[k.length];
        int[] topKCounts = new int[k.length];
        float[] topKMeans = new float[k.length];
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < k.length; j++) {
                if (i < k[j]) {
                    topKSum[j] += arr[i];
                    topKCounts[j]++;
                }
            }
        }
        for (int j = 0; j < k.length; j++) {
            if (topKCounts[j] > 0) {
                topKMeans[j] = topKSum[j] / topKCounts[j];
            }
        }
        return topKMeans;
    }

    public static HashMap<Integer, Integer[]> getIdxStarMap(final int[] impressions, final int[] stars) {
        HashMap<Integer, LinkedList<Integer>> idxStarsListMap = new HashMap<>();
        for (int star = 0; star < 6; star++) {
            idxStarsListMap.put(star, new LinkedList<>());
        }
        for (int i = 0; i < impressions.length; i++) {
            if (idxStarsListMap.get(stars[i]) == null) {
                idxStarsListMap.put(stars[i], new LinkedList<>());
            }
            idxStarsListMap.get(stars[i]).add(i);
        }
        HashMap<Integer, Integer[]> idxStarsMap = new HashMap<>();
        for (int star : idxStarsListMap.keySet()) {
            idxStarsMap.put(star, idxStarsListMap.get(star).toArray(new Integer[0]));
        }
        return idxStarsMap;
    }

    public static HashMap<Integer, Integer[]> getIdxRatingMap(final int[] impressions, final int[] ratings) {
        HashMap<Integer, LinkedList<Integer>> idxRatingsListMap = new HashMap<>();
        for (int rating = 0; rating < 5; rating++) {
            idxRatingsListMap.put(rating, new LinkedList<>());
        }
        for (int i = 0; i < impressions.length; i++) {
            if (idxRatingsListMap.get(ratings[i]) == null) {
                idxRatingsListMap.put(ratings[i], new LinkedList<>());
            }
            idxRatingsListMap.get(ratings[i]).add(i);
        }
        HashMap<Integer, Integer[]> idxRatingsMap = new HashMap<>();
        for (int rating : idxRatingsListMap.keySet()) {
            idxRatingsMap.put(rating, idxRatingsListMap.get(rating).toArray(new Integer[0]));
        }
        return idxRatingsMap;
    }

    public static HashMap<String, float[]> getPriceStatsStarsMap(HashMap<Integer, Integer[]> idxStarsMap, int[] prices, int[] priceRanking) {
        float[] priceMinStar = new float[6];
        float[] priceRankMinStar = new float[6];
        float[] priceAverageStar = new float[6];
        float[] priceRankAverageStar = new float[6];
        for (int star = 0; star < 6; star++) {
            Integer[] idxStars = idxStarsMap.get(star);
            int nItems = 0;
            float priceMin = 0F;
            float priceRankMin = 0F;
            float priceSum = 0F;
            float priceRankSum = 0F;
            for (int i = 0; i < idxStars.length; i++) {
                if (i == 0 || prices[idxStars[i]] < priceMin) {
                    priceMin = prices[idxStars[i]];
                    priceRankMin = priceRanking[idxStars[i]];
                }
                priceSum += prices[idxStars[i]];
                priceRankSum += priceRanking[idxStars[i]];
                nItems++;
            }
            if (nItems > 0) {
                priceMinStar[star] = priceMin;
                priceRankMinStar[star] = priceRankMin;
                priceAverageStar[star] = priceSum / (float) nItems;
                priceRankAverageStar[star] = priceRankSum / (float) nItems;
            }
        }
        HashMap<String, float[]> priceStatsStarsMap = new HashMap<>();
        priceStatsStarsMap.put("priceMinStar", priceMinStar);
        priceStatsStarsMap.put("priceRankMinStar", priceRankMinStar);
        priceStatsStarsMap.put("priceAverageStar", priceAverageStar);
        priceStatsStarsMap.put("priceRankAverageStar", priceRankAverageStar);
        return priceStatsStarsMap;
    }

    public static HashMap<String, float[]> getPriceStatsRatingsMap(HashMap<Integer, Integer[]> idxRatingsMap, int[] prices, int[] priceRanking) {
        float[] priceMinRating = new float[5];
        float[] priceRankMinRating = new float[5];
        float[] priceAverageRating = new float[5];
        float[] priceRankAverageRating = new float[5];
        for (int rating = 0; rating < 5; rating++) {
            Integer[] idxRatings = idxRatingsMap.get(rating);
            int nItems = 0;
            float priceMin = 0F;
            float priceRankMin = 0F;
            float priceSum = 0F;
            float priceRankSum = 0F;
            for (int i = 0; i < idxRatings.length; i++) {
                if (i == 0 || prices[idxRatings[i]] < priceMin) {
                    priceMin = prices[idxRatings[i]];
                    priceRankMin = priceRanking[idxRatings[i]];
                }
                priceSum += prices[idxRatings[i]];
                priceRankSum += priceRanking[idxRatings[i]];
                nItems++;
            }
            if (nItems > 0) {
                priceMinRating[rating] = priceMin;
                priceRankMinRating[rating] = priceRankMin;
                priceAverageRating[rating] = priceSum / (float) nItems;
                priceRankAverageRating[rating] = priceRankSum / (float) nItems;
            }
        }
        HashMap<String, float[]> priceStatsRatingsMap = new HashMap<>();
        priceStatsRatingsMap.put("priceMinRating", priceMinRating);
        priceStatsRatingsMap.put("priceRankMinRating", priceRankMinRating);
        priceStatsRatingsMap.put("priceAverageRating", priceAverageRating);
        priceStatsRatingsMap.put("priceRankAverageRating", priceRankAverageRating);
        return priceStatsRatingsMap;
    }

    public static HashMap<String, HashMap<Integer, Integer>> getStarRankingMaps(HashMap<Integer, Integer[]> idxStarsMap, int[] prices) {
        HashMap<Integer, Integer> rankStarsMap = new HashMap<>();
        HashMap<Integer, Integer> rankLocalStarsMap = new HashMap<>();
        HashMap<Integer, Integer> priceStarsMap = new HashMap<>();
        HashMap<Integer, Integer> priceRankLocalStarsMap = new HashMap<>();
        HashMap<Integer, Integer> priceRankLocalAboveStarsMap = new HashMap<>();
        for (int star = 0; star < 6; star++) {
            Integer[] idxsStar = idxStarsMap.get(star);
            int[] rankStars = new int[idxsStar.length];
            int[] rankLocalStars = new int[idxsStar.length];
            int[] priceStars = new int[idxsStar.length];
            for (int i = 0; i < idxsStar.length; i++) {
                rankStars[i] = idxsStar[i] + 1;
                rankLocalStars[i] = i + 1;
                priceStars[i] = prices[idxsStar[i]];
            }
            int[] priceRankLocalStars = RecSys19Helper.argsort(priceStars, true, true);
            int[] priceRankLocalAboveStars = RecSys19Helper.getPriceRankAbove(priceRankLocalStars);
            for (int i = 0; i < idxsStar.length; i++) {
                rankStarsMap.put(idxsStar[i], rankStars[i]);
                rankLocalStarsMap.put(idxsStar[i], rankLocalStars[i]);
                priceStarsMap.put(idxsStar[i], priceStars[i]);
                priceRankLocalStarsMap.put(idxsStar[i], priceRankLocalStars[i]);
                priceRankLocalAboveStarsMap.put(idxsStar[i], priceRankLocalAboveStars[i]);
            }
        }
        HashMap<String, HashMap<Integer, Integer>> starRankLocalMaps = new HashMap<>();
        starRankLocalMaps.put("rankLocalStars", rankLocalStarsMap);
        starRankLocalMaps.put("priceRankLocalStars", priceRankLocalStarsMap);
        starRankLocalMaps.put("priceRankLocalAboveStars", priceRankLocalAboveStarsMap);
        return starRankLocalMaps;
    }

    public static HashMap<String, HashMap<Integer, Integer>> getRatingRankingMaps(HashMap<Integer, Integer[]> idxRatingsMap, int[] prices) {
        HashMap<Integer, Integer> rankRatingsMap = new HashMap<>();
        HashMap<Integer, Integer> rankLocalRatingsMap = new HashMap<>();
        HashMap<Integer, Integer> priceRatingsMap = new HashMap<>();
        HashMap<Integer, Integer> priceRankLocalRatingsMap = new HashMap<>();
        HashMap<Integer, Integer> priceRankLocalAboveRatingsMap = new HashMap<>();
        for (int rating = 0; rating < 5; rating++) {
            Integer[] idxsRating = idxRatingsMap.get(rating);
            int[] rankRatings = new int[idxsRating.length];
            int[] rankLocalRatings = new int[idxsRating.length];
            int[] priceRatings = new int[idxsRating.length];
            for (int i = 0; i < idxsRating.length; i++) {
                rankRatings[i] = idxsRating[i] + 1;
                rankLocalRatings[i] = i + 1;
                priceRatings[i] = prices[idxsRating[i]];
            }
            int[] priceRankLocalRatings = RecSys19Helper.argsort(priceRatings, true, true);
            int[] priceRankLocalAboveRatings = RecSys19Helper.getPriceRankAbove(priceRankLocalRatings);
            for (int i = 0; i < idxsRating.length; i++) {
                rankRatingsMap.put(idxsRating[i], rankRatings[i]);
                rankLocalRatingsMap.put(idxsRating[i], rankLocalRatings[i]);
                priceRatingsMap.put(idxsRating[i], priceRatings[i]);
                priceRankLocalRatingsMap.put(idxsRating[i], priceRankLocalRatings[i]);
                priceRankLocalAboveRatingsMap.put(idxsRating[i], priceRankLocalAboveRatings[i]);
            }
        }
        HashMap<String, HashMap<Integer, Integer>> ratingRankLocalMaps = new HashMap<>();
        ratingRankLocalMaps.put("rankLocalRatings", rankLocalRatingsMap);
        ratingRankLocalMaps.put("priceRankLocalRatings", priceRankLocalRatingsMap);
        ratingRankLocalMaps.put("priceRankLocalAboveRatings", priceRankLocalAboveRatingsMap);
        return ratingRankLocalMaps;
    }

}