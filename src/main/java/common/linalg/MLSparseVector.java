package common.linalg;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class MLSparseVector implements Serializable {

    private static final long serialVersionUID = -8319046980055965552L;
    private int[] indexes;
    private float[] values;
    private long[] dates;
    private int length;

    public MLSparseVector(final int[] indexesP, final float[] valuesP,
                          final long[] datesP, final int lengthP) {
        int nnz = 0;

        if (indexesP != null && indexesP.length > 0) {
            this.indexes = indexesP;
            nnz = this.indexes.length;
        } else {
            this.indexes = null;
        }

        if (valuesP != null && valuesP.length > 0) {
            if (valuesP.length != nnz) {
                throw new IllegalArgumentException(
                        "indexes, values and dates must be of the same size");
            }
            this.values = valuesP;
        } else {
            this.values = null;
        }

        if (datesP != null && datesP.length > 0) {
            if (datesP.length != nnz) {
                throw new IllegalArgumentException(
                        "indexes, values and dates must be of the same size");
            }
            this.dates = datesP;
        } else {
            this.dates = null;
        }

        this.length = lengthP;
    }

    public void applyDateThresh(final long dateThresh, final boolean greater) {
        if (this.isEmpty() == true) {
            //nothing to do
            return;
        }

        // count how many dates are over the threshold
        int nPass = 0;
        for (int i = 0; i < this.dates.length; i++) {
            if ((greater == true && this.dates[i] > dateThresh)
                    || (greater == false && this.dates[i] <= dateThresh)) {
                nPass++;
            }
        }

        if (nPass == 0) {
            this.indexes = null;
            this.values = null;
            this.dates = null;
            return;
        }

        // apply date threshold
        int[] indexesThresh = new int[nPass];
        float[] valuesThresh = new float[nPass];
        long[] datesThresh = new long[nPass];

        int curIndex = 0;
        for (int j = 0; j < this.dates.length; j++) {
            if ((greater == true && this.dates[j] > dateThresh)
                    || (greater == false && this.dates[j] <= dateThresh)) {
                indexesThresh[curIndex] = this.indexes[j];
                valuesThresh[curIndex] = this.values[j];
                datesThresh[curIndex] = this.dates[j];
                curIndex++;
            }
        }

        this.indexes = indexesThresh;
        this.values = valuesThresh;
        this.dates = datesThresh;
    }

    public void applyIndexSelector(final Map<Integer, Integer> selectedIndexMap,
                                   final int nColsSelected) {
        if (this.isEmpty() == true) {
            this.length = nColsSelected;
            return;
        }


        if (nColsSelected == 0) {
            this.indexes = null;
            this.values = null;
            this.dates = null;
            this.length = nColsSelected;
            return;
        }

        // apply column selector in place to this vector
        List<MLMatrixElement> reindexElms = new ArrayList<MLMatrixElement>(
                this.indexes.length);
        for (int i = 0; i < this.indexes.length; i++) {
            Integer newIndex = selectedIndexMap.get(this.indexes[i]);
            if (newIndex != null) {
                if (this.hasDates() == true) {
                    reindexElms.add(new MLMatrixElement(-1, newIndex,
                            this.values[i], this.dates[i]));
                } else {
                    reindexElms.add(new MLMatrixElement(-1, newIndex,
                            this.values[i], -1));
                }
            }
        }

        if (reindexElms.size() == 0) {
            // nothing selected
            this.indexes = null;
            this.values = null;
            this.dates = null;
            this.length = nColsSelected;
            return;
        }

        Collections.sort(reindexElms,
                new MLMatrixElement.ColIndexComparator(false));
        int[] prunedIndexes = new int[reindexElms.size()];
        float[] prunedValues = new float[reindexElms.size()];
        long[] prunedDates = null;
        if (this.hasDates() == true) {
            prunedDates = new long[reindexElms.size()];
        }

        int cur = 0;
        for (MLMatrixElement element : reindexElms) {

            prunedIndexes[cur] = element.getColIndex();
            prunedValues[cur] = element.getValue();
            if (this.hasDates() == true) {
                prunedDates[cur] = element.getDate();
            }
            cur++;
        }

        this.indexes = prunedIndexes;
        this.values = prunedValues;
        this.dates = prunedDates;
        this.length = nColsSelected;
    }

    public void applyNorm(final int p) {
        float rowNorm = this.getNorm(p);
        if (rowNorm < 1e-5f) {
            return;
        }
        this.divide(rowNorm);
    }

    public void applyNorm(final MLDenseVector norm) {
        if (this.isEmpty() == true) {
            //nothing to do
            return;
        }

        if (this.length != norm.getLength()) {
            throw new IllegalArgumentException("length != length");
        }

        float[] normValues = norm.getValues();
        for (int i = 0; i < this.indexes.length; i++) {
            if (normValues[this.indexes[i]] > 1e-10f) {
                this.values[i] /= normValues[this.indexes[i]];
            }
        }
    }

    public MLSparseVector deepCopy() {
        int[] indexesClone = null;
        if (this.indexes != null) {
            indexesClone = this.indexes.clone();
        }

        float[] valuesClone = null;
        if (this.values != null) {
            valuesClone = this.values.clone();
        }

        long[] datesClone = null;
        if (this.dates != null) {
            datesClone = this.dates.clone();
        }

        return new MLSparseVector(indexesClone, valuesClone,
                datesClone, this.length);
    }

    public void divide(final float constant) {
        if (this.isEmpty() == true) {
            //nothing to do
            return;
        }

        for (int i = 0; i < this.values.length; i++) {
            this.values[i] /= constant;
        }
    }

    public long[] getDates() {
        return this.dates;
    }

    public int[] getIndexes() {
        return this.indexes;
    }

    public int getLength() {
        return this.length;
    }

    public float getNorm(final int p) {
        if (this.isEmpty() == true) {
            //nothing to do
            return 0f;
        }

        double rowNorm = 0;
        for (int i = 0; i < this.values.length; i++) {
            if (p == 1) {
                rowNorm += Math.abs(this.values[i]);
            } else {
                rowNorm += Math.pow(this.values[i], p);
            }
        }
        if (p != 1) {
            rowNorm = Math.pow(rowNorm, 1.0 / p);
        }

        return (float) rowNorm;
    }

    public float[] getValues() {
        return this.values;
    }

    public boolean hasDates() {
        return this.dates != null;
    }


    public int intersect(final MLSparseVector other) {
        if (this.isEmpty() == true || other.isEmpty() == true) {
            return 0;
        }

        if (this.length != other.length) {
            throw new IllegalArgumentException("length != length");
        }

        int maxIndex = this.indexes[this.indexes.length - 1];
        if (other.getIndexes()[0] > maxIndex) {
            // no overlap in indexes
            return 0;
        }

        int intersect = 0;
        int[] otherIndexes = other.getIndexes();

        int cur = 0;
        int curOther = 0;
        while (true) {
            if (cur >= this.length || curOther >= otherIndexes.length) {
                break;
            }

            if (otherIndexes[curOther] > maxIndex) {
                // indexes are sorted so can exit here
                break;
            }

            if (this.indexes[cur] == otherIndexes[curOther]) {
                intersect++;
                cur++;
                curOther++;

            } else if (this.indexes[cur] > otherIndexes[curOther]) {
                curOther++;

            } else {
                cur++;
            }
        }

        return intersect;
    }

    public boolean isEmpty() {
        return this.indexes == null;
    }

    public float max() {
        if (this.isEmpty() == true) {
            return 0f;
        }

        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < this.values.length; i++) {
            if (max < this.values[i]) {
                max = this.values[i];
            }
        }
        return max;
    }

    public void merge(final MLSparseVector vecToMerge) {
        if (this.getLength() != vecToMerge.getLength()) {
            throw new IllegalArgumentException(
                    "vector lengths must be the same to merge");
        }
        if (vecToMerge.isEmpty() == true) {
            return;
        }
        if (this.isEmpty() == true) {
            this.indexes = vecToMerge.getIndexes();
            this.values = vecToMerge.getValues();
            this.dates = vecToMerge.getDates();
            return;
        }

        boolean hasDates = this.hasDates();

        Map<Integer, MLMatrixElement> rowMap = new TreeMap<Integer,
                MLMatrixElement>();
        for (int i = 0; i < vecToMerge.getIndexes().length; i++) {
            if (hasDates == true) {
                rowMap.put(vecToMerge.getIndexes()[i],
                        new MLMatrixElement(1, vecToMerge.getIndexes()[i],
                                vecToMerge.getValues()[i],
                                vecToMerge.getDates()[i]));
            } else {
                rowMap.put(vecToMerge.getIndexes()[i],
                        new MLMatrixElement(1, vecToMerge.getIndexes()[i],
                                vecToMerge.getValues()[i], 0L));
            }
        }

        for (int i = 0; i < this.indexes.length; i++) {
            MLMatrixElement element = rowMap.get(this.indexes[i]);
            if (element == null) {
                if (hasDates == true) {
                    rowMap.put(this.indexes[i], new MLMatrixElement(1,
                            this.indexes[i], this.values[i], this.dates[i]));
                } else {
                    rowMap.put(this.getIndexes()[i], new MLMatrixElement(1,
                            this.indexes[i], this.values[i], 0L));
                }
            } else {
                if (hasDates == true) {
                    if (this.dates[i] > element.getDate()) {
                        // store most recent date
                        element.setDate(this.dates[i]);
                    }
                }
                // sum up values
                element.setValue(element.getValue() + this.getValues()[i]);
            }
        }

        int[] indexesMerged = new int[rowMap.size()];
        float[] valuesMerged = new float[rowMap.size()];
        long[] datesMerged = null;
        if (hasDates == true) {
            datesMerged = new long[rowMap.size()];
        }

        int index = 0;
        for (Map.Entry<Integer, MLMatrixElement> entry : rowMap.entrySet()) {
            MLMatrixElement element = entry.getValue();
            indexesMerged[index] = element.getColIndex();
            valuesMerged[index] = element.getValue();
            if (hasDates == true) {
                datesMerged[index] = element.getDate();
            }
            index++;
        }

        this.indexes = indexesMerged;
        this.values = valuesMerged;
        this.dates = datesMerged;
    }

    public float min() {
        if (this.isEmpty() == true) {
            //nothing to do
            return 0f;
        }

        float min = Float.POSITIVE_INFINITY;
        for (int i = 0; i < this.values.length; i++) {
            if (min > this.values[i]) {
                min = this.values[i];
            }
        }
        return min;
    }

    public float multiply(final MLSparseVector other) {
        if (this.length != other.length) {
            throw new IllegalArgumentException("length != length");
        }

        if (this.isEmpty() == true || other.isEmpty() == true) {
            return 0f;
        }

        int maxIndex = this.indexes[this.indexes.length - 1];
        if (other.getIndexes()[0] > maxIndex) {
            // no overlap in indexes
            return 0f;
        }

        double product = 0f;
        int[] otherIndexes = other.getIndexes();
        float[] otherValues = other.getValues();

        int cur = 0;
        int curOther = 0;
        while (true) {
            if (cur >= this.indexes.length || curOther >= otherIndexes.length) {
                break;
            }

            if (otherIndexes[curOther] > maxIndex) {
                // indexes are sorted so can exit here
                break;
            }

            if (this.indexes[cur] == otherIndexes[curOther]) {
                product += this.values[cur] * otherValues[curOther];
                cur++;
                curOther++;

            } else if (this.indexes[cur] > otherIndexes[curOther]) {
                curOther++;

            } else {
                cur++;
            }
        }

        return (float) product;
    }

    public void setDates(long[] datesP) {
        if (datesP != null && datesP.length == 0) {
            this.dates = null;
        } else {
            this.dates = datesP;
        }
    }

    public void setIndexes(int[] indexesP) {
        if (indexesP != null && indexesP.length == 0) {
            this.indexes = null;
        } else {
            this.indexes = indexesP;
        }
    }

    public void setLength(int dim) {
        this.length = dim;
    }

    public void setValues(float[] valuesP) {
        if (valuesP != null && valuesP.length == 0) {
            this.values = null;
        } else {
            this.values = valuesP;
        }
    }

    public MLSparseVector subtract(final MLSparseVector other) {
        if (this.getLength() != other.getLength()) {
            throw new IllegalArgumentException(
                    "vectors must have equall lengths");
        }

        if (this.isEmpty() == true) {
            if (other.isEmpty() == true) {
                return new MLSparseVector(null, null, null, this.getLength());
            } else {
                float[] values = other.getValues();
                for (int i = 0; i < values.length; i++) {
                    values[i] = -values[i];
                }
                return new MLSparseVector(other.indexes.clone(), values, null
                        , this.getLength());
            }
        }

        if (other.isEmpty() == true) {
            return this.deepCopy();
        }

        float[] result = new float[this.getLength()];

        for (int i = 0; i < this.indexes.length; i++) {
            result[this.indexes[i]] += this.values[i];
        }

        int[] otherIndexes = other.getIndexes();
        float[] otherValues = other.getValues();
        for (int i = 0; i < otherIndexes.length; i++) {
            result[otherIndexes[i]] -= otherValues[i];
        }
        return new MLDenseVector(result).toSparse();
    }

    public MLDenseVector toDense() {
        float[] dense = new float[this.getLength()];
        if (this.isEmpty() == true) {
            return new MLDenseVector(dense);
        }

        for (int i = 0; i < this.indexes.length; i++) {
            dense[this.indexes[i]] = this.values[i];
        }
        return new MLDenseVector(dense);
    }

    public String toLIBSVMString(int offset) {

        if (this.isEmpty() == true) {
            return "";
        }

        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < this.indexes.length; i++) {
            float val = this.values[i];
            if (val == Math.round(val)) {
                builder.append(
                        " " + (offset + this.indexes[i]) + ":" + ((int) val));
            } else {
                builder.append(" " + (offset + this.indexes[i]) + ":"
                        + String.format("%.5f", val));
            }
        }
        return builder.toString();
    }

    public static MLSparseVector concat(final MLSparseVector... vectors) {
        int length = 0;
        int nnz = 0;
        boolean copyDates = true;
        for (int i = 0; i < vectors.length; i++) {
            MLSparseVector vector = vectors[i];
            if (vector.getIndexes() != null) {
                nnz += vector.getIndexes().length;
            }
            length += vectors[i].getLength();
            if (vector.isEmpty() == false && vector.hasDates() == false) {
                // all vectors must have dates to concat
                copyDates = false;
            }
        }
        int[] indexes = new int[nnz];
        float[] values = new float[nnz];
        long[] dates = null;
        if (copyDates == true) {
            dates = new long[nnz];
        }
        int cur = 0;
        int offset = 0;
        for (int i = 0; i < vectors.length; i++) {
            MLSparseVector vector = vectors[i];
            int[] vecInds = vector.getIndexes();
            if (vecInds != null) {
                float[] vecVals = vector.getValues();
                long[] vecDates = vector.getDates();
                for (int j = 0; j < vecInds.length; j++) {
                    indexes[cur] = offset + vecInds[j];
                    values[cur] = vecVals[j];
                    if (copyDates == true) {
                        dates[cur] = vecDates[j];
                    }
                    cur++;
                }
            }
            offset += vector.getLength();
        }
        return new MLSparseVector(indexes, values, dates, length);
    }

    public static MLSparseVector fromDense(final MLDenseVector dense) {
        float[] denseVals = dense.getValues();

        int nnz = 0;
        for (int i = 0; i < denseVals.length; i++) {
            if (denseVals[i] != 0) {
                nnz++;
            }
        }
        if (nnz == 0) {
            return new MLSparseVector(null, null, null, denseVals.length);
        }

        int[] indexes = new int[nnz];
        float[] values = new float[nnz];
        int cur = 0;
        for (int i = 0; i < denseVals.length; i++) {
            if (denseVals[i] != 0) {
                indexes[cur] = i;
                values[cur] = denseVals[i];
                cur++;
            }
        }
        return new MLSparseVector(indexes, values, null, denseVals.length);
    }

    public static MLDenseVector mean(final MLSparseVector... input) {
        int n = input.length;

        if (n == 0) {
            throw new IllegalArgumentException(
                    "Can't average over no vectors.");
        }

        int d = input[0].getLength();
        for (int i = 0; i < n; i++) {
            if (input[i].getLength() != d) {
                throw new IllegalArgumentException("Vector at position " + i
                        + " has length " + input[i].length
                        + " but first vector had length " + d + ".");
            }
        }

        double[] sum = new double[d];

        for (int i = 0; i < n; i++) {
            if (input[i].isEmpty() == true) {
                continue;
            }
            for (int j = 0; j < input[i].indexes.length; j++) {
                int index = input[i].indexes[j];
                float value = input[i].values[j];
                sum[index] += value;
            }
        }

        float[] result = new float[sum.length];
        for (int i = 0; i < d; i++) {
            result[i] = (float) (sum[i] / n);
        }

        return new MLDenseVector(result);
    }

}
