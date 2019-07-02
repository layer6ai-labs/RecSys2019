package common.eval;

import java.io.Serializable;

public class MLEvalResult implements Comparable<MLEvalResult>, Serializable {
	private static final long serialVersionUID = 22998468127105885L;
	private String objective;
	private double[] result;
	private double[][] resultRows;
	private int nEval;

	public MLEvalResult(final String objectiveP,
						final double[] resultP,
						final int nEvalP) {
		this.objective = objectiveP;
		this.result = resultP;
		this.resultRows = null;
		this.nEval = nEvalP;
	}

	public MLEvalResult(final String objectiveP,
						final double[] resultP,
						final double[][] resultRowsP,
						final int nEvalP) {
		this.objective = objectiveP;
		this.result = resultP;
		this.resultRows = resultRowsP;
		this.nEval = nEvalP;
	}

	@Override
	public int compareTo(final MLEvalResult o) {
		return Double.compare(this.last(), o.last());
	}

	public double[] get() {
		return this.result;
	}

	public double[][] getRows() {
		return this.resultRows;
	}

	public String getObjective() {
		return this.objective;
	}

	public int getNEval() {
		return this.nEval;
	}

	private double last() {
		return this.result[this.result.length - 1];
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("nEval: " + this.nEval + ", ");
		builder.append(this.objective + ":");
		for (int i = 0; i < this.result.length; i++) {
			builder.append(String.format(" %.4f", this.result[i]));
		}
		return builder.toString();
	}
}
