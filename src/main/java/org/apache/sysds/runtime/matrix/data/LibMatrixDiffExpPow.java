package org.apache.sysds.runtime.matrix.data;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class LibMatrixDiffExpPow {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    /**
     * Raises each element in a DoubleVector by a certain power.
     *
     * @param vec    The input DoubleVector.
     * @param power  The power to raise each element to.
     * @return DoubleVector with each element raised to the specified power.
     */
    private static DoubleVector power(DoubleVector vec, double power) {
        return vec.pow(power);
    }

    /**
     * Applies the exponential function to each element in a DoubleVector.
     *
     * @param vec The input DoubleVector.
     * @return DoubleVector with exp() applied on each element.
     */
    private static DoubleVector exp(DoubleVector vec) {
        return vec.lanewise(VectorOperators.EXP);
    }

    /**
     * Computes the difference between two DoublesVectors.
     *
     * @param vec1 The first DoubleVector.
     * @param vec2 The second DoubleVector.
     * @return DoubleVector that is the result of: vec2 - vec1
     */
    private static DoubleVector diff(DoubleVector vec1, DoubleVector vec2) {
        return vec2.sub(vec1);
    }
//
//    /**
//     * Raises each element of the given matrix to the specified power using vectorization.
//     *
//     * @param matrix The input matrix.
//     * @param power  The power to raise each element to.
//     * @return A new matrix with each element raised to the specified power.
//     */
//    public static double[][] powerMatrix(MatrixBlock matrix, double power) {
//        int rows = matrix.length;
//        if (rows == 0) return new double[0][0];
//        int cols = matrix[0].length;
//
//        double[][] result = new double[rows][cols];
//
//        for (int i = 0; i < rows; i++) {
//            int j = 0;
//            // Vectorized processing
//            for (; j <= SPECIES.loopBound(cols); j += SPECIES.length()) {
//                if (j + SPECIES.length() <= cols) {
//                    // Load elements into a vector
//                    DoubleVector vector = DoubleVector.fromArray(SPECIES, matrix[i], j);
//                    // Raise each element to the specified power
//                    DoubleVector powered = power(vector, power);
//                    // Store the result back into the result matrix
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
//    public static MatrixBlock expMatrix(MatrixBlock mb) {
//
//        int rows = mb.rlen;;
//        int cols = mb.clen;
//
//        double[][] result = new double[rows][cols];
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
//                DoubleVector res = exp(vector);
//                res.intoArray(mb.getDenseBlockValues(), j);
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
//    public static double[][] diffMatrix(MatrixBlock matrix) {
//        int rows = matrix.length;
//        if (rows == 0) return new double[0][0];
//        int cols = matrix[0].length;
//
//        // Ensure that each row has at least two columns
//        for (int i = 0; i < rows; i++) {
//            if (matrix[i].length < 2) {
//                throw new IllegalArgumentException("All rows must have at least two columns to compute differences.");
//            }
//        }
//
//        int diffCols = cols - 1;
//        double[][] result = new double[rows][diffCols];
//
//        for (int i = 0; i < rows; i++) {
//            int j = 0;
//            // Vectorized processing
//            for (; j <= SPECIES.loopBound(diffCols); j += SPECIES.length()) {
//                if (j + SPECIES.length() <= diffCols) {
//                    // Load current elements starting at j
//                    DoubleVector currentVector = DoubleVector.fromArray(SPECIES, matrix[i], j);
//                    // Load next elements starting at j + 1
//                    DoubleVector nextVector = DoubleVector.fromArray(SPECIES, matrix[i], j + 1);
//                    // Compute the difference: next - current
//                    DoubleVector diffVector = diff(currentVector, nextVector);
//                    // Store the result back into the result matrix
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
