package org.apache.sysds.performance.matrix;

import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult2;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

public class MatrixMulSIMDPerformance {

    private static final String BASE_PATH = "vector_api_test/";

    public static void simdMultTests(double sparsityMatrix1, double sparsityMatrix2, int k, String stepStr) {
        // Define the file path for the CSV output
        String outputPath = BASE_PATH + "performance_" + sparsityMatrix1 + "_" + sparsityMatrix2 + "_k=" + k + ".csv";
        long startTime1 = 0, endTime1 = 0, startTime2 = 0, endTime2 = 0;
        long avg1 = 0, avg2 = 0;

        try (FileWriter writer = new FileWriter(outputPath)) {
            // Write CSV header
            writer.append("n,exec_time_no_simd,exec_time_simd\n");

            // densedenseMM multiplication
            int[] sizes = calculateSizes(stepStr);
            for (int n : sizes) {
                // Generate two random dense matrices
                MatrixBlock mbA = MatrixBlock.randOperations(n, n, sparsityMatrix1, 0, 1, "uniform", 7);
                MatrixBlock mbB = MatrixBlock.randOperations(n, n, sparsityMatrix2, 0, 1, "uniform", 7);

                // Measure the execution time of the matrix multiplication
                avg1 = 0;
                avg2 = 0;

                for(int i = -5; i < 10; i++) {
                    startTime1 = System.nanoTime();
                    LibMatrixMult.matrixMult(mbA, mbB, k); // No SIMD
                    endTime1 = System.nanoTime();

                    if(i >= 0) {
                        avg1 += (endTime1 - startTime1) / 1000000;
                    }
                }

                for(int i = -5; i < 10; i++) {
                    startTime2 = System.nanoTime();
                    LibMatrixMult2.matrixMult(mbA, mbB, k); // SIMD
                    endTime2 = System.nanoTime();

                    if(i >= 0) {
                        avg2 += (endTime2 - startTime2) / 1000000;
                    }
                }

                System.out.println("------------------");
                System.out.println("Averages - NO-SIMD: " + avg1/10 + "; SIMD: avg2/10");
                // Write to csv
                //writer.append(n + "," + avg1/10 + "," + avg2/10 + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static int[] calculateSizes(String stepStr) {
        int n = 0;
        String[] strArr = stepStr.split("-");
        String[] strArr2 = strArr[1].split("#");
        int start = Integer.valueOf(strArr[0]);
        int end = Integer.valueOf(strArr2[0]);
        int step = Integer.valueOf(strArr2[1]);

        for(int i = start; i <= end; i+=step) {
            n++;
        }

        int[] sizes = new int[n];
        for(int i = 0; i < sizes.length; i++) {
            sizes[i] = start+step*i;
        }

        return sizes;
    }
}
