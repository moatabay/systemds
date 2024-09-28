package org.apache.sysds.runtime.matrix.data;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.apache.sysds.runtime.data.DenseBlock;

public class LibMatrixDiffExpPow {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    /**
     * Raises each element of the given matrix to the specified power using vectorization.
     *
     * @param matrix The input matrix.
     * @param power  The power to raise each element to.
     * @return A new matrix with each element raised to the specified power.
     */
//    public static MatrixBlock powerMatrix(MatrixBlock matrix, double power) {
//        int rows = matrix.getNumRows();
//        int cols = matrix.getNumColumns();
//
//        MatrixBlock ret = new MatrixBlock();
//        DenseBlock block = matrix.getDenseBlock();
//
//        for (int i = 0; i < rows; i++) {
//            int j = 0;
//            // Vectorized processing
//            for (; j <= SPECIES.loopBound(cols); j += SPECIES.length()) {
//                if (j + SPECIES.length() <= cols) {
//                    DoubleVector vector = DoubleVector.fromArray(SPECIES, matrix[i], j);
//                    DoubleVector powered = vector.lanewise(VectorOperators.POW, power); // Like that or without the "DoubleVector powered ="
//                    powered.intoArray(result[i], j);
//                }
//            }
//            // Handle remaining elements
//            for (; j < cols; j++) {
//                result[i][j] = Math.pow(matrix[i][j], power);
//            }
//        }
//
//        return result;
//    }
//
//    /**
//     * Applies the exponential function to each element of the given matrix using vectorization.
//     *
//     * @param mb The input matrix.
//     * @return A new matrix with the exponential of each element.
//     */
//    public static MatrixBlock expMatrix(MatrixBlock matrix) {
//
//        int rows = matrix.getNumRows();
//        int cols = matrix.getNumColumns();
//
//        MatrixBlock ret = new MatrixBlock();
//        int speciesLen = SPECIES.length();
//        int max = SPECIES.loopBound(cols);
//
//        for (int i = 0; i < rows; i++) {
//            int j = 0;
//            // Rest
//            for (; j < cols % speciesLen; j++) {
//                result[i][j] = Math.exp(matrix[i][j]);
//            }
//
//            // Block-wise iteration
//            for (; j <= max; j += speciesLen) {
//                DoubleVector vector = DoubleVector.fromArray(SPECIES, mb.getDenseBlockValues(), j);
//                DoubleVector res = vector.lanewise(VectorOperators.EXP);
//                res.intoArray(matrix.getDenseBlockValues(), j);
//            }
//        }
//
//        return result;
//    }
//
//    /**
//     * Computes the difference between consecutive elements in each row of the given matrix using vectorization.
//     * Specifically, for each row, it calculates diff[j] = row[j+1] - row[j].
//     *
//     * @param matrix The input matrix.
//     * @return A new matrix where each row contains the differences between consecutive elements of the input matrix.
//     * @throws IllegalArgumentException If any row in the matrix has fewer than two columns.
//     */
//    public static double[][] diffMatrix(MatrixBlock matrix1, MatrixBlock matrix2) {
//        if(matrix1.getNumRows() != matrix2.getNumRows() || matrix1.getNumColumns() != matrix2.getNumColumns()) {
//            // TODO: Throw Exception
//        }
//        int rows1 = matrix1.getNumRows();
//        int cols1 = matrix1.getNumColumns();
//        int rows2 = matrix2.getNumRows();
//        int cols2 = matrix2.getNumColumns();
//
//        /* ???????????????????
//        // Ensure that each row has at least two columns
//        for (int i = 0; i < rows; i++) {
//            if (matrix[i].length < 2) {
//                throw new IllegalArgumentException("All rows must have at least two columns to compute differences.");
//            }
//        }*/
//
//        int diffCols = cols - 1;
//        MatrixBlock ret = new MatrixBlock();
//
//        for (int i = 0; i < rows; i++) {
//            int j = 0;
//            // Vectorized processing
//            for (; j <= SPECIES.loopBound(diffCols); j += SPECIES.length()) {
//                if (j + SPECIES.length() <= diffCols) {
//                    DoubleVector currentVector = DoubleVector.fromArray(SPECIES, matrix[i], j);
//                    DoubleVector nextVector = DoubleVector.fromArray(SPECIES, matrix[i], j + 1);
//                    DoubleVector diffVector = vec2.sub(vec1);
//                    diffVector.intoArray(result[i], j);
//                }
//            }
//            // Handle remaining elements
//            for (; j < diffCols; j++) {
//                result[i][j] = matrix[i][j + 1] - matrix[i][j];
//            }
//        }
//
//        return result;
//    }

}
