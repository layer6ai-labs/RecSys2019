package recsys2019;
import common.feature.MLSparseFeature;
import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class RecSys19Data implements Serializable {
    private static final long serialVersionUID = 1L;
    public static final String CLICKOUT_ITEM_ACTION = "clickout item";
    public static final String INTERACTION_ITEM_RATING_ACTION = "interaction item rating";
    public static final String INTERACTION_ITEM_INFO_ACTION = "interaction item info";
    public static final String INTERACTION_ITEM_IMAGE_ACTION = "interaction item image";
    public static final String INTERACTION_ITEM_DEALS_ACTION = "interaction item deals";
    public static final String CHANGE_OF_SORT_ORDER_ACTION = "change of sort order";
    public static final String FILTER_SELECTION_ACTION = "filter selection";
    public static final String SEARCH_FOR_ITEM_ACTION = "search for item";
    public static final String SEARCH_FOR_DESTINATION_ACTION = "search for destination";
    public static final String SEARCH_FOR_POI_ACTION = "search for poi";
    public static final long VALID_SPLIT_END = 1541548799;
    public static final long VALID_SPLIT_START = VALID_SPLIT_END - 50_000;

    public enum SessionFeature {
        user_id,
        session_id,
        timestamp,
        step,
        action_type,
        reference,
        reference_sort_order,
        reference_filter,
        reference_search_dest,
        reference_search_poi,
        platform,
        city,
        device,
        current_filters,
        impressions,
        prices;
        public static int getColumnIndex(final String[] header, final SessionFeature featureName) {
            for (int i = 0; i < header.length; i++) {
                if (header[i].equals(featureName.name()) == true) {
                    return i;
                }
            }
            throw new IllegalStateException(featureName.name() + " not found in " + Arrays.toString(header));
        }
    }

    public enum ItemFeature {
        item_id,
        properties;
        public static int getColumnIndex(final String[] header, final ItemFeature featureName) {
            for (int i = 0; i < header.length; i++) {
                if (header[i].equals(featureName.name()) == true) {
                    return i;
                }
            }
            throw new IllegalStateException(featureName.name() + " not found in " + Arrays.toString(header));
        }
    }

    public Map<Integer, Integer> getIndexToItemId() {
        Map<Integer, Integer> indexToItemId = new HashMap(itemIdToIndex.size());
        for (Map.Entry<Integer, Integer> entry : itemIdToIndex.entrySet()) {
            indexToItemId.put(entry.getValue(), entry.getKey());
        }
        return indexToItemId;
    }

    public int[] referenceItems;
    public long[] timeStamps;
    public int[][] impressions;
    public int[][] prices;
    public Map<Integer, Integer> itemIdToIndex;
    public Map<ItemFeature, MLSparseFeature> itemFeatures;
    public Map<Integer, Set<Integer>> userToSessionStart;
    public Map<Integer, Set<Integer>> trainToSessionStart;
    public Map<Integer, Set<Integer>> testToSessionStart;
    public Map<SessionFeature, MLSparseFeature> sessionFeatures;
    public int[] trainEventIndexes; //sorted
    public int[] validEventIndexes; //sorted
    public int[] testEventIndexes; //sorted

    public RecSys19Data() {

    }
}
