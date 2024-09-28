package org.apache.sysds.runtime.matrix.data;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;

public class LibMatrixDiffExpPow2 {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    /**
     * Computes the element-wise difference between two matrices.
     * @param A MatrixBlock A
     * @param B MatrixBlock B
     * @param ret The resulting matrix
     */
    public static void diff(MatrixBlock A, MatrixBlock B, MatrixBlock ret) {
        int numRows = A.getNumRows();
        int numCols = A.getNumColumns();

        // Check for dimensions compatibility
        if (A.getNumRows() != B.getNumRows() || A.getNumColumns() != B.getNumColumns()) {
            throw new IllegalArgumentException("Matrix dimensions do not match.");
        }

        DenseBlock a = A.getDenseBlock();
        DenseBlock b = B.getDenseBlock();
        DenseBlock c = ret.getDenseBlock();

        for (int r = 0; r < numRows; r++) {
            double[] aRow = a.valuesAt(r);
            double[] bRow = b.valuesAt(r);
            double[] outputRow = c.valuesAt(r);

            for (int col = 0; col < numCols; col += SPECIES.length()) {
                DoubleVector va = DoubleVector.fromArray(SPECIES, aRow, col);
                DoubleVector vb = DoubleVector.fromArray(SPECIES, bRow, col);
                DoubleVector vresult = va.sub(vb); // A - B
                vresult.intoArray(outputRow, col);
            }
        }
    }

    /**
     * Computes the element-wise exponential for a matrix.
     * @param A Input MatrixBlock
     * @param ret The resulting matrix
     */
    public static void exp(MatrixBlock A, MatrixBlock ret) {
        int numRows = A.getNumRows();
        int numCols = A.getNumColumns();

        for (int r = 0; r < numRows; r++) {
            double[] aRow = new double[]{A.getDenseBlockValues()[r]};
            double[] outputRow = new double[]{ret.getDenseBlockValues()[r]};

            for (int c = 0; c < numCols; c += SPECIES.length()) {
                var va = DoubleVector.fromArray(SPECIES, aRow, c);
                var vresult = va.lanewise(VectorOperators.EXP); // exp(A)
                vresult.intoArray(outputRow, c);
            }
        }
    }

    /**
     * Computes the element-wise power for a matrix.
     * @param A Input MatrixBlock
     * @param exponent Exponent to raise each element to
     * @param ret The resulting matrix
     */
    public static void pow(MatrixBlock A, double exponent, MatrixBlock ret) {
        int numRows = A.getNumRows();
        int numCols = A.getNumColumns();

        for (int r = 0; r < numRows; r++) {
            double[] aRow = new double[]{A.getDenseBlockValues()[r]};
            double[] outputRow = new double[]{ret.getDenseBlockValues()[r]};

            for (int c = 0; c < numCols; c += SPECIES.length()) {
                var va = DoubleVector.fromArray(SPECIES, aRow, c);
                var vresult = va.lanewise(VectorOperators.POW, exponent); // A^exponent
                vresult.intoArray(outputRow, c);
            }
        }
    }
}
