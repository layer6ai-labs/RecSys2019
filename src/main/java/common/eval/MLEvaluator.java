package common.eval;

import common.linalg.FloatElement;
import common.linalg.MLSparseMatrix;

public abstract class MLEvaluator {

	protected int[] threshs;

	public MLEvaluator() {
	}

	public MLEvaluator(final int[] threshsP) {
		this.threshs = threshsP;
	}

	/**
	 * Evaluation function, each row of targets contains ground truth indexes
	 * and each row of preds contains sorted predictions. Null rows in both
	 * targets and preds are skipped. Note that for metrics such as non-binary
	 * NDCG, target relevances can be put as values into the target matrix.
	 * 
	 * @param targets
	 *            each row of targets contains ground truth indexes
	 * @param preds
	 *            each row of preds contains sorted predictions
	 * @return
	 */
	public abstract MLEvalResult evaluate(final MLSparseMatrix targets,
			final FloatElement[][] preds);

	public abstract String getName();

	public int[] getEvalThreshs() {
		return this.threshs;
	}

	public int getMaxEvalThresh() {
	    int maxThresh = -1;
	    for (int i = 0; i < this.threshs.length; i++) {
	        if (this.threshs[i] > maxThresh) {
	            maxThresh = this.threshs[i];
            }
        }
        return maxThresh;
	}

	public int getMaxEvalThresh(int[] threshsP) {
		int maxThresh = -1;
		for (int i = 0; i < threshsP.length; i++) {
			if (threshsP[i] > maxThresh) {
				maxThresh = threshsP[i];
			}
		}
		return maxThresh;
	}

}
