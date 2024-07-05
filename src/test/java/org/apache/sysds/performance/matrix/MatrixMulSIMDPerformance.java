package org.apache.sysds.performance.matrix;

import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult2;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.FileWriter;
import java.io.IOException;

public class MatrixMulSIMDPerformance {

    private static int[] sizes = {1000, 5000, 10000, 15000};
    private static final String BASE_PATH = "vector_api_test/";

    public static void simdMultTests(double sparsityMatrix1, double sparsityMatrix2, int k) {
        System.gc();
        // Define the file path for the CSV output
        String outputPath = BASE_PATH + "performance_" + sparsityMatrix1 + "_" + sparsityMatrix2 + "_k=" + k + ".csv";
        long startTime1 = 0, endTime1 = 0, startTime2 = 0, endTime2 = 0;
        long avg1 = 0, avg2 = 0;

        try (FileWriter writer = new FileWriter(outputPath)) {
            // Write CSV header
            writer.append("n,exec_time_no_simd,exec_time_simd\n");

            // densedenseMM multiplication
            for (int n : sizes) {
                // Generate two random dense matrices
                MatrixBlock mbA = MatrixBlock.randOperations(n, n, sparsityMatrix1, 0, 1, "uniform", 7);
                MatrixBlock mbB = MatrixBlock.randOperations(n, n, sparsityMatrix2, 0, 1, "uniform", 7);

                // Measure the execution time of the matrix multiplication
                avg1 = 0;
                avg2 = 0;

                for(int i = -8; i < 10; i++) {
                    startTime1 = System.nanoTime();
                    LibMatrixMult.matrixMult(mbA, mbB, k); // No SIMD
                    endTime1 = System.nanoTime();

                    if(i >= 0) {
                        avg1 += (endTime1 - startTime1) / 1000000;
                    }
                }

                for(int i = -8; i < 10; i++) {
                    startTime2 = System.nanoTime();
                    LibMatrixMult2.matrixMult(mbA, mbB, k); // SIMD
                    endTime2 = System.nanoTime();

                    if(i >= 0) {
                        avg2 += (endTime2 - startTime2) / 1000000;
                    }
                }

                // Write to csv
                writer.append(n + "," + avg1/10 + "," + avg2/10 + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
