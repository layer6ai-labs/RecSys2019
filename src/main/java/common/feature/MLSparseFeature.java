package common.feature;

import common.linalg.*;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Class for common feature extraction.
 * To create:
 * 1. initialize
 * 2. addRow() to inserted data
 * 3. finalizeFeature() once all data is inserted
 * 4. getRowTransformed() to get feature rows
 * <p>
 * To serialize:
 * 1. prepareToSerialize() if called with true data will also be saved
 * 2. serialize feature object using MLIOUtils.writeObjectToFile()
 * 3. finishSerialize()
 * <p>
 * To load from serialized:
 * 1. load using MLIOUtils.readObjectFromFile()
 * 2. if saved without data:
 * - call prepareForData()
 * - load data with addRow()
 * - finalizeFeature() once all data is inserted
 * <p>
 * To use in batch mode:
 * 1.load first batch and call finalizeFeature(), this will compute and
 * fix all transforms
 * 2.for each new batch:
 * - call clearData() to remove data from feature
 * - load batch data with addRow()
 * - call finalizeFeature(), note that transforms will not be
 * re-calculated so this procedure will produce consistent data
 */
public class MLSparseFeature implements Serializable {

    private static final long serialVersionUID = 4665530401588164620L;

    private MLSparseMatrix featMatrix;
    private MLSparseMatrix featMatrixTransformed;
    private Map<String, Integer> catToIndex;
    private AtomicInteger curCatIndex;
    // NOTE: last transform in this sequence must tokenize text
    private MLTextTransform[] textTransforms;
    private MLFeatureTransform[] featTransforms;
    private AtomicBoolean inInfMode;
    private int nCols;
    private Class<? extends MLSparseMatrix> type;

    private transient MLSparseMatrix featMatrixCache;
    private transient MLSparseMatrix featMatrixTransCache;

    public <T extends MLSparseMatrix> MLSparseFeature(final int nRowsP,
                                                      final MLTextTransform[] textTransformsP,
                                                      final MLFeatureTransform[] featTransformsP, final Class<T> typeP) {
        this.type = typeP;

        this.catToIndex = new HashMap<>();
        this.curCatIndex = new AtomicInteger(-1);

        this.textTransforms = textTransformsP;
        if (this.textTransforms != null && this.textTransforms.length == 0) {
            this.textTransforms = null;
        }

        this.featTransforms = featTransformsP;
        if (this.featTransforms != null && this.featTransforms.length == 0) {
            this.featTransforms = null;
        }

        this.inInfMode = new AtomicBoolean(false);

        this.prepareForData(nRowsP);
    }

    public <T extends MLSparseMatrix> MLSparseFeature(final int nRowsP,
                                                      final MLTextTransform[] textTransformsP,
                                                      final MLFeatureTransform[] transformsP, final Class<T> typeP,
                                                      final MLSparseFeature anotherFeature) {
        this(nRowsP, textTransformsP, transformsP, typeP);

        // share category maps with another feature
        this.catToIndex = anotherFeature.catToIndex;
        this.curCatIndex = anotherFeature.curCatIndex;
    }

    public <T extends MLSparseMatrix> MLSparseFeature(
            final MLTextTransform[] textTransformsP,
            final MLFeatureTransform[] transformsP,
            final MLSparseMatrix featMatrixP) {
        this(featMatrixP.getNRows(), textTransformsP, transformsP,
                featMatrixP.getClass());

        // init feature with existing data matrix
        this.featMatrix = featMatrixP;
    }


    /**
     * Makes an 'inference version' of this MLSparseFeature, without any data.
     * Method finalize must have first been called.
     * <p>
     * This copies over the catMaps, number of columns, and feature transforms,
     * and prepares the feature to receive data.
     * <p>
     * This should have the same effect as the serialization / deserialization
     * the is normally carried out between training and inference.
     *
     * @param nRows          Number of rows to be added to this inference
     *                       version.
     * @param anotherFeature MLSparseFeature from which cat maps and feature
     *                       transforms will be taken.
     */
    public MLSparseFeature(final int nRows,
                           final MLSparseFeature anotherFeature) {

        // Pass anotherFeature to the constructor to ensure that we get the
        // catMaps copied over.
        this(nRows, anotherFeature.textTransforms,
                anotherFeature.featTransforms, anotherFeature.type,
                anotherFeature);

        // Guard against doing this with a non-finalized feature.
        if (anotherFeature.inInfMode.get() == false) {
            throw new IllegalStateException("MLSparseFeature must be " +
                    "finalized before using this constructor");
        }

        // Need to do this because when
        // we call finalize after reading data,
        // this MLSparseFeature will already be inInfMode.
        this.nCols = anotherFeature.nCols;

        // Because anotherFeature sets transforms
        this.inInfMode.set(true);
    }

    public void addRow(final int rowIndex, final float value) {
        if (value == 0) {
            this.featMatrix.setRow(null, rowIndex);
            return;
        }

        if (this.type.equals(MLSparseMatrixFlat.class) == true) {
            ((MLSparseMatrixFlat) this.featMatrix).setRow(0, value, rowIndex);

        } else {
            this.featMatrix.setRow(new MLSparseVector(new int[]{0},
                    new float[]{value}, null, this.nCols), rowIndex);
        }
    }

    public void addRow(final int rowIndex, final MLDenseVector dense) {
        if (this.inInfMode.get() == true && dense.getLength() != this.featMatrix.getNCols()) {
            throw new IllegalArgumentException("in inference mode vector " +
                    "length must match feature matrix nCols");
        }

        MLSparseVector sparse = dense.toSparse();
        if (sparse.isEmpty() == false) {
            this.featMatrix.setRow(sparse, rowIndex);
        } else {
            this.featMatrix.setRow(null, rowIndex);
        }
    }

    public void addRow(final int rowIndex, final MLSparseVector sparse) {
        if (this.inInfMode.get() == true && sparse.getLength() != this.featMatrix.getNCols()) {
            throw new IllegalArgumentException("in inference mode vector " +
                    "length must match feature matrix nCols");
        }

        if (sparse.isEmpty() == false) {
            this.featMatrix.setRow(sparse, rowIndex);
        } else {
            this.featMatrix.setRow(null, rowIndex);
        }
    }

    public void addRow(final int rowIndex, final String text) {

        if (this.textTransforms == null) {
            // no transforms so treat as category
            Integer index = this.getCatIndex(text);
            if (index == null) {
                this.featMatrix.setRow(null, rowIndex);
                return;
            }

            if (this.type.equals(MLSparseMatrixFlat.class) == true) {
                // can set row directly here
                ((MLSparseMatrixFlat) this.featMatrix).setRow(index, 1,
                        rowIndex);
            } else {
                this.featMatrix.setRow(new MLSparseVector(new int[]{index},
                                new float[]{1}, null, this.catToIndex.size()),
                        rowIndex);
            }

        } else {
            // apply transforms and tokenize
            MLTextTransform.MLTextInput input =
                    new MLTextTransform.MLTextInput(text);
            for (MLTextTransform inputTransform : this.textTransforms) {
                inputTransform.apply(input);
            }

            String[] tokenized = input.getTokenized();
            if (tokenized != null && tokenized.length > 0) {
                this.addRow(rowIndex, tokenized);
            } else {
                this.featMatrix.setRow(null, rowIndex);
            }
        }
    }

    public void addRow(final int rowIndex, final String[] cats) {
        addRow(rowIndex, cats, null);
    }

    public void addRow(final int rowIndex, final String[] cats,
                       final float[] values) {
        // map values to indexes and add sparse row to matrix
        MLSparseVector sparse = this.getFeatVector(cats, values);
        if (sparse.isEmpty() == false) {
            this.featMatrix.setRow(sparse, rowIndex);
        } else {
            this.featMatrix.setRow(null, rowIndex);
        }
    }

    /**
     * Clear data from this feature. This function can be called
     * repeatedly in batch mode to load new data.
     */
    public void clearData() {
        this.featMatrix.clearData();
    }


    public void sliceRows(final int fromIndex, final int toIndex) {
        // only keep rows in [fromIndex, toIndex)
        this.featMatrix = this.featMatrix.sliceRows(fromIndex, toIndex);
    }

    public synchronized void finalizeFeature(final boolean preserveOrig) {
        // NOTE: this fn must be called before feature can be used

        if (this.inInfMode.get() == false) {
            // not in inference mode so infer nCols
            this.featMatrix.inferAndSetNCols();

            // this is necessary for features with shared cat maps
            int nColsCat = 0;
            for (Integer index : this.catToIndex.values()) {
                if (nColsCat < (index + 1)) {
                    nColsCat = index + 1;
                }
            }
            if (this.featMatrix.getNCols() < nColsCat) {
                this.featMatrix.setNCols(nColsCat);
            }

            //NOTE nCols can't be changed after this
            this.nCols = this.featMatrix.getNCols();
        } else {
            //in inference mode so set nCols
            this.featMatrix.setNCols(this.nCols);
        }

        // apply all transforms
        if (preserveOrig == true) {
            // deep copy
            this.featMatrixTransformed = this.featMatrix.deepCopy();
        } else {
            // shallow copy
            this.featMatrixTransformed = this.featMatrix;
        }

        if (this.featTransforms != null) {
            for (MLFeatureTransform transform : this.featTransforms) {
                if (this.inInfMode.get() == true) {
                    // don't recalculate transforms in inf mode
                    transform.applyInference(this);
                } else {
                    transform.apply(this);
                }
            }
        }
        this.inInfMode.set(true);
    }

    public synchronized void finishSerialize() {
        // NOTE: must call this after serialization
        if (this.featMatrix == null) {
            this.featMatrix = this.featMatrixCache;
            this.featMatrixTransformed = this.featMatrixTransCache;
        }
    }

    private synchronized Integer getCatIndex(final String cat) {
        Integer index = this.catToIndex.get(cat);
        if (index == null) {
            if (this.inInfMode.get() == false) {
                index = this.curCatIndex.incrementAndGet();
                this.catToIndex.put(cat, index);
            }
        }
        return index;
    }

    public Map<String, Integer> getCatToIndex() {
        return this.catToIndex;
    }

    public MLSparseVector getFeatInf(final float value) {
        if (this.inInfMode.get() == false) {
            throw new IllegalStateException(
                    "feature is not in inference mode, call finalizeFeature()");
        }

        if (value == 0) {
            return new MLSparseVector(null,
                    null, null, 1);
        }
        MLSparseVector vector = new MLSparseVector(new int[]{0},
                new float[]{value}, null, 1);

        // apply feature transforms
        if (this.featTransforms != null) {
            for (MLFeatureTransform transform : this.featTransforms) {
                transform.applyInference(vector);
            }
        }
        return vector;
    }

    public MLSparseVector getFeatInf(final MLDenseVector dense) {
        if (this.inInfMode.get() == false) {
            throw new IllegalStateException(
                    "feature is not in inference mode, call finalizeFeature()");
        }

        if (this.nCols != dense.getLength()) {
            throw new IllegalArgumentException(
                    "this.nCols != dense.getLength()");
        }

        // map dense to sparse
        MLSparseVector vector = dense.toSparse();

        // apply feature transforms
        if (this.featTransforms != null) {
            for (MLFeatureTransform transform : this.featTransforms) {
                transform.applyInference(vector);
            }
        }
        return vector;
    }

    public MLSparseVector getFeatInf(final MLSparseVector sparse) {
        if (this.inInfMode.get() == false) {
            throw new IllegalStateException(
                    "feature is not in inference mode, call finalizeFeature()");
        }

        if (this.nCols != sparse.getLength()) {
            throw new IllegalArgumentException(
                    "this.nCols != sparse.getLength()");
        }

        // apply feature transforms
        if (this.featTransforms != null) {
            for (MLFeatureTransform transform : this.featTransforms) {
                transform.applyInference(sparse);
            }
        }
        return sparse;
    }

    public MLSparseVector getFeatInf(final String text) {
        if (this.inInfMode.get() == false) {
            throw new IllegalStateException(
                    "feature is not in inference mode, call finalizeFeature()");
        }

        MLSparseVector vector = null;
        if (this.textTransforms == null) {
            // no text transforms so treat this as category
            vector = this.getFeatVector(text);
        } else {

            // apply text transforms
            MLTextTransform.MLTextInput input =
                    new MLTextTransform.MLTextInput(text);
            for (MLTextTransform inputTransform : this.textTransforms) {
                inputTransform.apply(input);
            }

            // map tokenized text to sparse vector
            String[] tokenized = input.getTokenized();
            vector = this.getFeatVector(tokenized, null);
        }

        // apply feature transforms
        if (this.featTransforms != null) {
            for (MLFeatureTransform transform : this.featTransforms) {
                transform.applyInference(vector);
            }
        }
        return vector;
    }

    public MLSparseVector getFeatInf(final String[] cats) {
        return getFeatInf(cats, null);
    }

    public MLSparseVector getFeatInf(final String[] cats,
                                     final float[] values) {
        if (this.inInfMode.get() == false) {
            throw new IllegalStateException(
                    "feature is not in inference mode, call finalizeFeature()");
        }

        // map categories to sparse
        MLSparseVector vector = this.getFeatVector(cats, values);

        // apply all the transforms
        if (this.featTransforms != null) {
            for (MLFeatureTransform transform : this.featTransforms) {
                transform.applyInference(vector);
            }
        }
        return vector;
    }

    public MLSparseMatrix getFeatMatrix() {
        return this.featMatrix;
    }

    public MLSparseMatrix getFeatMatrixTransformed() {
        return this.featMatrixTransformed;
    }

    public String[] getFeatNames(final String prefix,
                                 final boolean transformed) {
        if (this.nCols == 0) {
            return new String[0];
        }

        if (this.catToIndex.size() == 0) {
            // numerical feature
            String[] featNames = new String[]{prefix};
            if (transformed == true && this.featTransforms != null) {
                // get feature names after all transforms are applied
                for (MLFeatureTransform transform : this.featTransforms) {
                    featNames = transform.applyFeatureName(featNames);
                }
            }
            return featNames;
        }

        // get feature name in the format 'prefix_[cat name]'
        String[] featNames = new String[this.catToIndex.size()];
        for (Map.Entry<String, Integer> entry : this.catToIndex.entrySet()) {
            featNames[entry.getValue()] = prefix + "_" + entry.getKey().trim()
                    .replaceAll("\\s+", "_");
        }

        if (transformed == true && this.featTransforms != null) {
            // get feature names after all transforms are applied
            for (MLFeatureTransform transform : this.featTransforms) {
                featNames = transform.applyFeatureName(featNames);
            }
        }

        return featNames;
    }

    private MLSparseVector getFeatVector(final String cat) {
        Integer index = this.getCatIndex(cat);
        if (index == null) {
            return new MLSparseVector(null, null, null, this.catToIndex.size());
        } else {
            return new MLSparseVector(new int[]{index}, new float[]{1},
                    null, this.catToIndex.size());
        }
    }

    private MLSparseVector getFeatVector(final String[] cats,
                                         final float[] catValues) {
        if (catValues != null && cats.length != catValues.length) {
            throw new IllegalArgumentException("cats and catValues length do " +
                    "not match!");
        }
        TreeMap<Integer, MutableFloat> countMap = new TreeMap<>();
        for (int i = 0; i < cats.length; i++) {
            Integer index = this.getCatIndex(cats[i]);
            if (index == null) {
                continue;
            }

            MutableFloat count = countMap.get(index);
            if (count == null) {
                if (catValues == null) {
                    countMap.put(index, new MutableFloat(1));
                } else {
                    countMap.put(index, new MutableFloat(catValues[i]));
                }
            } else {
                //aggregate counts for repeated categories
                if (catValues == null) {
                    count.value++;
                } else {
                    count.value += catValues[i];
                }
            }
        }

        if (countMap.size() == 0) {
            return new MLSparseVector(null, null, null, this.nCols);
        }

        int[] indexes = new int[countMap.size()];
        float[] values = new float[countMap.size()];
        int cur = 0;
        for (Map.Entry<Integer, MutableFloat> entry : countMap.entrySet()) {
            indexes[cur] = entry.getKey();
            values[cur] = entry.getValue().value;
            cur++;
        }

        // NOTE this.nCols can be wrong in non-inference
        // mode since we don't know number of categories ahead of
        // time. This is later corrected by calling finalizeFeature().
        return new MLSparseVector(indexes, values, null, this.nCols);
    }

    public Map<Integer, String> getIndexToCat() {
        Map<Integer, String> indexToCat = new HashMap<>(this.catToIndex.size());
        for (Map.Entry<String, Integer> entry : this.catToIndex.entrySet()) {
            indexToCat.put(entry.getValue(), entry.getKey());
        }
        return indexToCat;
    }

    public MLSparseVector getRow(final int rowIndex,
                                 final boolean returnEmpty) {
        return this.featMatrix.getRow(rowIndex, returnEmpty);
    }

    public MLSparseVector getRowTransformed(final int rowIndex,
                                            final boolean returnEmpty) {
        if (this.inInfMode.get() == false) {
            throw new IllegalStateException(
                    "feature is not in inference mode, call finalizeFeature()");
        }

        return this.featMatrixTransformed.getRow(rowIndex, returnEmpty);
    }

    public boolean infMode() {
        return this.inInfMode.get();
    }

    public synchronized void prepareForData(final int nRows) {
        if (this.type.equals(MLSparseMatrixAOO.class) == true) {
            if (this.inInfMode.get() == true) {
                this.featMatrix = new MLSparseMatrixAOO(nRows, this.nCols);
            } else {
                this.featMatrix = new MLSparseMatrixAOO(nRows, 0);
            }

        } else if (this.type.equals(MLSparseMatrixFlat.class) == true) {
            if (this.inInfMode.get() == true) {
                this.featMatrix = new MLSparseMatrixFlat(nRows, this.nCols);
            } else {
                this.featMatrix = new MLSparseMatrixFlat(nRows, 0);
            }

        } else {
            throw new IllegalArgumentException(
                    "unsupported type " + this.type.getName());
        }
    }

    public synchronized void prepareToSerialize(final boolean withData) {
        if (withData == false) {
            // NOTE: this excludes data from serialization so
            // all data loading must be re-run and you must call
            // finalize before using this feature after de-serialization.
            this.featMatrixCache = this.featMatrix;
            this.featMatrixTransCache = this.featMatrixTransformed;

            this.featMatrix = null;
            this.featMatrixTransformed = null;
            // NOTE: must call finishSerialize() if the data in this
            // feature is to be used after serialization.
        }
    }

    /**
     * Use existing category mapping to initialize this feature.
     *
     * @param catToIndexP
     */
    public synchronized void setCatMap(final Map<String, Integer> catToIndexP) {
        this.catToIndex = new HashMap<>(catToIndexP.size());
        this.curCatIndex = new AtomicInteger(0);
        for (Map.Entry<String, Integer> entry : catToIndexP.entrySet()) {
            this.catToIndex.put(entry.getKey(), entry.getValue());
            if (this.curCatIndex.get() < entry.getValue()) {
                this.curCatIndex.set(entry.getValue());
            }
        }
    }
}