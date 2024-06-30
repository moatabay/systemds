package org.apache.sysds.performance.matrix;

import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult2;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.FileWriter;
import java.io.IOException;

public class MatrixMulSIMDPerformance {

    private static int[] sizes = {1000, 2000, 3000, 15000};
    private static final String BASE_PATH = "vector_api_test/";

    public static void simdMultTests(double sparsityMatrix1, double sparsityMatrix2) {
        // Define the file path for the CSV output
        String outputPath = BASE_PATH + "performance_" + sparsityMatrix1 + "_" + sparsityMatrix2 + ".csv";
        long startTime1 = 0, endTime1 = 0, startTime2 = 0, endTime2 = 0, duration1 = 0, duration2 = 0;

        try (FileWriter writer = new FileWriter(outputPath)) {
            // Write CSV header
            writer.append("n,exec_time_no_simd,exec_time_simd\n");

            // densedenseMM multiplication
            for (int n : sizes) {
                // Generate two random dense matrices
                MatrixBlock mbA = MatrixBlock.randOperations(n, n, sparsityMatrix1, 0, 1, "uniform", 7);
                MatrixBlock mbB = MatrixBlock.randOperations(n, n, sparsityMatrix2, 0, 1, "uniform", 7);

                // Measure the execution time of the matrix multiplication (no SIMD)
                startTime1 = System.nanoTime();
                LibMatrixMult.matrixMult(mbA, mbB);
                endTime1 = System.nanoTime();
                duration1 = (endTime1 - startTime1) / 1000000;

                startTime2 = System.nanoTime();
                LibMatrixMult2.matrixMult(mbA, mbB);
                endTime2 = System.nanoTime();
                duration2 = (endTime2 - startTime2) / 1000000;

                // Write to csv
                writer.append(n + "," + duration1 + "," + duration2 + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
