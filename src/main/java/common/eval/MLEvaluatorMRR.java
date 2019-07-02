package common.eval;

import common.linalg.FloatElement;
import common.linalg.MLSparseMatrix;
import common.linalg.MLSparseVector;
import com.google.common.util.concurrent.AtomicDouble;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class MLEvaluatorMRR extends MLEvaluator {

    public MLEvaluatorMRR() {
        super(null);
    }

    @Override
    public MLEvalResult evaluate(final MLSparseMatrix targets,
                                 final FloatElement[][] preds) {

        AtomicDouble mrr = new AtomicDouble(0);
        AtomicInteger nTotal = new AtomicInteger(0);
        IntStream.range(0, preds.length).parallel().forEach(index -> {

            MLSparseVector row = targets.getRow(index, false);
            FloatElement[] rowPreds = preds[index];

            if (row == null || rowPreds == null) {
                // skip if null rows are encountered
                return;
            }

            nTotal.incrementAndGet();
            int[] indexes = row.getIndexes();
            for (int i = 0; i < rowPreds.length; i++) {
                if (Arrays.binarySearch(indexes,
                        rowPreds[i].getIndex()) >= 0) {
                    mrr.addAndGet(1.0 / (1.0 + i));
                    break;
                }
            }
        });

        return new MLEvalResult(this.getName(),
                new double[]{mrr.get() / nTotal.get()}, nTotal.get());
    }

    @Override
    public String getName() {
        return "MRR";
    }
}
