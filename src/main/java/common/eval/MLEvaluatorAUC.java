package common.eval;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import common.linalg.FloatElement;
import common.linalg.MLSparseMatrix;
import common.linalg.MLSparseVector;
import com.google.common.util.concurrent.AtomicDouble;

/**
 * AUC (1-vs-all):
 * - null class rows or edges cases are either skipped or thrown an exception
 * - degenerate sample scores are assigned group midpoint interpolated rank
 */
public class MLEvaluatorAUC extends MLEvaluator {

    private boolean outputPerRow = false;

    public MLEvaluatorAUC() {
        super(null);
    }

    public MLEvaluatorAUC(final boolean outputPerRowP) {
        super(null);
        this.outputPerRow = outputPerRowP;
    }

    @Override
    public MLEvalResult evaluate(final MLSparseMatrix targets,
                                 final FloatElement[][] preds) {

        AtomicDouble auc = new AtomicDouble(0);

        double[][] aucsRows;
        if (this.outputPerRow) {
            aucsRows = new double[preds.length][1];
        } else {
            aucsRows = null;
        }

        AtomicInteger nTotal = new AtomicInteger(0);
        IntStream.range(0, preds.length).parallel().forEach(index -> {

            MLSparseVector row = targets.getRow(index, false);
            FloatElement[] rowPreds = preds[index];

            if (row == null || rowPreds == null) {
                // skip if null rows are encountered
                return;
            }

            int[] targetIndexes = row.getIndexes(); // positive sample indexes

            int pos = targetIndexes.length; // num positive samples
            int neg = rowPreds.length - targetIndexes.length; // num negative
            // samples

            if (pos == 0 || neg == 0) {
                // can't eval so skip
                return;
            }

            // Assign rank for each sample
            double[] rank = new double[rowPreds.length];
            for (int i = 0; i < rowPreds.length; i++) {
                if (i == rowPreds.length - 1 || rowPreds[i].getValue() != rowPreds[i + 1].getValue()) {
                    // majority of samples -> assign true rank
                    rank[i] = i + 1;
                } else {
                    // degenerate samples -> assign average group rank
                    int j = i + 1;
                    while (j < rowPreds.length && rowPreds[j].getValue() == rowPreds[i].getValue()) {
                        j++;
                    }
                    double r = (i + 1 + j) / 2.0; // rank midpoint interpolation
                    for (int k = i; k < j; k++) {
                        rank[k] = r;
                    }
                    i = j - 1;
                }
            }

            // Sum ranks of negative samples only
            double rankSum = 0.0;
            for (int i = 0; i < rowPreds.length; i++) {
                if (Arrays.binarySearch(targetIndexes,
                        rowPreds[i].getIndex()) < 0) {
                    rankSum += rank[i];
                }
            }

            // Compute per-row auc
            double curAuc =
                    (rankSum - (neg * (neg + 1.0) / 2.0)) / (neg * (double) pos);

            // Optionall fill in per-row aucs
            if (this.outputPerRow == true) {
                aucsRows[index][0] = curAuc; // per-row
            }

            // Sum up per-row aucs
            auc.addAndGet(curAuc);
            nTotal.incrementAndGet();
        });

        int nEval = Math.max(nTotal.get(), 1);
        return new MLEvalResult(this.getName(),
                new double[]{auc.get() / nEval}, aucsRows, nEval);
    }

    @Override
    public String getName() {
        return "auc";
    }
}
