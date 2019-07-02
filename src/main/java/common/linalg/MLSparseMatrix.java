package common.linalg;

import java.io.Serializable;
import java.util.Map;
import java.util.stream.IntStream;

public interface MLSparseMatrix extends Serializable {

    public abstract void applyColNorm(final MLDenseVector colNorm);

    public abstract void applyColSelector(
            final Map<Integer, Integer> selectedColMap,
            final int nColsSelected);

    public abstract void applyRowNorm(final MLDenseVector rowNorm);

    public abstract void binarizeValues();

    public abstract void clearData();

    public abstract MLSparseMatrix deepCopy();

    public abstract long[] getColNNZ();

    public abstract MLDenseVector getColNorm(final int p);

    public abstract MLDenseVector getColSum();

    public abstract int getNCols();

    public abstract long getNNZ();

    public abstract int getNRows();

    /**
     * @deprecated use getRow(rowIndex, false) to get the
     * same behavior
     */
    @Deprecated
    public abstract MLSparseVector getRow(final int rowIndex);

    /**
     * If returnEmpty=false and row at rowIndex is empty then
     * this function must return null. Otherwise if returnEmpty=true it must
     * always return a valid (possibly empty) object of MLSparseVector with
     * length properly set.
     *
     * @param rowIndex
     * @param returnEmpty
     * @return
     */
    public abstract MLSparseVector getRow(final int rowIndex,
                                          final boolean returnEmpty);

    public abstract long[] getRowNNZ();

    public abstract MLDenseVector getRowNorm(final int p);

    public abstract MLDenseVector getRowSum();

    public abstract boolean hasDates();

    public abstract MLSparseMatrix sliceRows(final int fromIndex,
                                             final int toIndex);

    public abstract MLSparseMatrix sliceRows(final int[] rowIndexes);

    public abstract void inferAndSetNCols();

    public abstract MLSparseMatrix mult(final MLSparseMatrix another);

    public abstract MLDenseVector multCol(final MLDenseVector vector);

    public abstract MLDenseVector multCol(final MLSparseVector vector);

    public abstract MLDenseVector multRow(final MLDenseVector vector);

    public abstract MLDenseVector multRow(final MLSparseVector vector);

    public abstract Map<Integer, Integer> selectCols(final int nnzCutOff);

    public abstract void setNCols(int nCols);

    public abstract void setRow(final MLSparseVector row, final int rowIndex);

    public abstract void toBinFile(final String outFile) throws Exception;

    public abstract MLSparseMatrix transpose();

    public static MLSparseMatrix concatHorizontal(
            final MLSparseMatrix... matrices) {
        int nRows = matrices[0].getNRows();
        int nColsNew = 0;
        for (MLSparseMatrix matrix : matrices) {
            if (nRows != matrix.getNRows()) {
                throw new IllegalArgumentException(
                        "input must have same number of rows");
            }
            nColsNew += matrix.getNCols();
        }

        MLSparseVector[] concat = new MLSparseVector[nRows];
        IntStream.range(0, nRows).parallel().forEach(rowIndex -> {

            MLSparseVector[] rows = new MLSparseVector[matrices.length];
            for (int i = 0; i < matrices.length; i++) {
                rows[i] = matrices[i].getRow(rowIndex, true);
            }
            MLSparseVector concatRow = MLSparseVector.concat(rows);
            if (concatRow.isEmpty() == false) {
                concat[rowIndex] = concatRow;
            }
        });

        return new MLSparseMatrixAOO(concat, nColsNew);
    }

    public static MLSparseMatrix concatVertical(
            final MLSparseMatrix... matrices) {

        int nCols = matrices[0].getNCols();
        int nRowsNew = 0;
        int[] offsets = new int[matrices.length];
        boolean[] hasDates = new boolean[]{true};
        for (int i = 0; i < offsets.length; i++) {
            if (nCols != matrices[i].getNCols()) {
                throw new IllegalArgumentException(
                        "input must have same number of columns");
            }
            nRowsNew += matrices[i].getNRows();
            offsets[i] = nRowsNew;

            if (matrices[i].hasDates() == false) {
                hasDates[0] = false;
            }
        }

        MLSparseVector[] concatRows = new MLSparseVector[nRowsNew];
        IntStream.range(0, nRowsNew).parallel().forEach(rowIndex -> {

            int offsetMatIndex = 0;
            int offsetRowIndex = 0;
            for (int i = 0; i < offsets.length; i++) {
                if (rowIndex < offsets[i]) {
                    offsetMatIndex = i;
                    if (i == 0) {
                        offsetRowIndex = rowIndex;
                    } else {
                        offsetRowIndex = rowIndex - offsets[i - 1];
                    }
                    break;
                }
            }

            MLSparseVector row = matrices[offsetMatIndex]
                    .getRow(offsetRowIndex);
            if (row != null) {
                concatRows[rowIndex] = row.deepCopy();
                if (hasDates[0] == false) {
                    // NOTE: if at least one matrix doesn't have dates
                    // then all dates must be removed
                    concatRows[rowIndex].setDates(null);
                }
            }
        });

        return new MLSparseMatrixAOO(concatRows, nCols);
    }
}
