package common.xgb;

import common.eval.MLEvalResult;
import common.eval.MLEvaluator;
import common.linalg.FloatElement;
import common.linalg.MLSparseMatrix;
import common.linalg.MLSparseMatrixAOO;
import common.linalg.MLSparseVector;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.IEvaluation;

import java.util.Arrays;
import java.util.stream.IntStream;

public class XGBEvaluator implements IEvaluation {

    private static final long serialVersionUID = 1282033966281920326L;

    private MLEvaluator evaluator;
    private boolean rowOriented;
    private int nClass;
    private boolean printFull;

    public XGBEvaluator(final MLEvaluator evaluatorP,
                        final boolean rowOrientedP,
                        final int nClassP,
                        final boolean printFullP) {
        this.evaluator = evaluatorP;
        this.rowOriented = rowOrientedP;
        this.nClass = nClassP;
        this.printFull = printFullP;
    }

    @Override
    public float eval(float[][] preds, final DMatrix dmat) {
        try {
            if (this.nClass == 2 && preds[0].length != 1) {
                throw new IllegalArgumentException(
                        "this.nClass == 2 && preds[0].length != 1: must use " +
                                "binary:logistic objective for binary " +
                                "classification");
            }

            if (this.nClass != 2 && this.nClass != preds[0].length) {
                throw new IllegalArgumentException("this.nClass != preds[0]" +
                        ".length");
            }

            float[] targets = dmat.getLabel();
            MLSparseMatrix targetsMat = this.targetsToMatrix(targets,
                    this.nClass, this.rowOriented);
            FloatElement[][] predsElement = this.predsToFloatElement(preds,
                    this.rowOriented);

            MLEvalResult result = this.evaluator.evaluate(targetsMat,
                    predsElement);
            if (this.printFull == true) {
                System.out.println(result.toString());
            }
            return (float) result.get()[0];

        } catch (Exception e) {
            e.printStackTrace();
            return -1;
        }
    }

    public float eval(float[][] preds, final float[] targets) {
        try {
            if (this.nClass == 2 && preds[0].length != 1) {
                throw new IllegalArgumentException(
                        "this.nClass == 2 && preds[0].length != 1: must use " +
                                "binary:logistic objective for binary " +
                                "classification");
            }

            if (this.nClass != 2 && this.nClass != preds[0].length) {
                throw new IllegalArgumentException("this.nClass != preds[0]" +
                        ".length");
            }

            MLSparseMatrix targetsMat = this.targetsToMatrix(targets,
                    this.nClass, this.rowOriented);
            FloatElement[][] predsElement = this.predsToFloatElement(preds,
                    this.rowOriented);

            MLEvalResult result = this.evaluator.evaluate(targetsMat,
                    predsElement);
            if (this.printFull == true) {
                System.out.println(result.toString());
            }
            return (float) result.get()[0];

        } catch (Exception e) {
            e.printStackTrace();
            return -1;
        }
    }

    @Override
    public String getMetric() {
        return this.evaluator.getName();
    }

    /**
     * Convert predictions to FloatElement, transpose (if doing column oriented
     * eval) and sort.
     */
    public static FloatElement[][] predsToFloatElement(final float[][] preds,
                                                       final boolean rowOriented) {
        FloatElement[][] predElements;
        if (rowOriented == true) {
            predElements = new FloatElement[preds.length][preds[0].length];
        } else {
            predElements = new FloatElement[preds[0].length][preds.length];
        }

        for (int i = 0; i < preds.length; i++) {
            for (int j = 0; j < preds[0].length; j++) {
                if (rowOriented == true) {
                    predElements[i][j] = new FloatElement(j, preds[i][j]);
                } else {
                    predElements[j][i] = new FloatElement(i, preds[i][j]);
                }
            }
        }

        IntStream.range(0, predElements.length).parallel().forEach(index -> {
            Arrays.sort(predElements[index],
                    new FloatElement.ValueComparator(true));
        });

        return predElements;
    }

    /**
     * Convert labels into matrix and transpose if doing column oriented eval.
     */
    public static MLSparseMatrix targetsToMatrix(final float[] labels,
                                                 final int nClass,
                                                 final boolean rowOriented) {
        MLSparseVector[] rows = new MLSparseVector[labels.length];

        IntStream.range(0, labels.length).parallel().forEach(index -> {
            int classIndex = (int) labels[index];
            if (classIndex < nClass) {
                if (nClass == 2) {
                    if (classIndex == 1) {
                        //treat 2-class problems as 1D
                        rows[index] = new MLSparseVector(new int[]{0},
                                new float[]{1}, null, nClass - 1);
                    }
                } else {
                    rows[index] = new MLSparseVector(new int[]{classIndex},
                            new float[]{1}, null, nClass);
                }
            }
        });

        MLSparseMatrix targetMatrix;
        if (nClass == 2) {
            targetMatrix = new MLSparseMatrixAOO(rows, nClass - 1);
        } else {
            targetMatrix = new MLSparseMatrixAOO(rows, nClass);
        }

        if (rowOriented == false) {
            targetMatrix = targetMatrix.transpose();
        }

        return targetMatrix;
    }
}