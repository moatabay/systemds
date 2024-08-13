package org.apache.sysds.performance.matrix;

import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult2;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.FileWriter;
import java.io.IOException;

import java.nio.file.Paths;
import java.nio.file.Files;

public class MatrixMulSIMDPerformance {

    private static final String BASE_PATH = "vector_api_test/";

    public static void simdMultTestsStaticKDynamicN(double sparsityMatrix1, double sparsityMatrix2, int k, String stepStr) {
        // Define the file path for the CSV output
        String outputPath = BASE_PATH + "performance1_" + sparsityMatrix1 + "_" + sparsityMatrix2 + "_k=" + k + ".csv";
        long startTime1 = 0, endTime1 = 0, startTime2 = 0, endTime2 = 0;
        long avg1 = 0, avg2 = 0;

        try {
            Files.createDirectories(Paths.get(BASE_PATH));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        try (FileWriter writer = new FileWriter(outputPath)) {
            // Write CSV header
            writer.append("n,k,exec_time_no_simd,exec_time_simd\n");

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

                System.out.println("Averages - NO-SIMD: " + (double) avg1/10 + "; SIMD: " + (double) avg2/10);
                System.out.println("------------------");
                // Write to csv
                writer.append(n + "," + k + "," + avg1/10 + "," + avg2/10 + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // dynamic k dynamic n
    // dynamic k static n
    // static k dynamic n
    // static k static n

    public static void simdMultTestsDynamicKStaticN(double sparsityMatrix1, double sparsityMatrix2, int n, String stepStr) {
        // Define the file path for the CSV output
        String outputPath = BASE_PATH + "performance2_" + sparsityMatrix1 + "_" + sparsityMatrix2 + "_n=" + n + ".csv";
        long startTime1 = 0, endTime1 = 0, startTime2 = 0, endTime2 = 0;
        long avg1 = 0, avg2 = 0;

        try {
            Files.createDirectories(Paths.get(BASE_PATH));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        try (FileWriter writer = new FileWriter(outputPath)) {
            // Write CSV header
            writer.append("n,k,exec_time_no_simd,exec_time_simd\n");

            int[] threadAmount = calculateSizes(stepStr);
            for (int k : threadAmount) {
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

                System.out.println("Averages - NO-SIMD: " + (double) avg1/10
                                                + "; SIMD: " + (double) avg2/10
                                                + "; n=" + n
                                                + "; k=" + k);
                System.out.println("------------------");
                // Write to csv
                writer.append(n + "," + k + "," + avg1/10 + "," + avg2/10 + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     *
     * @param stepStr
     * @return sizes-Array [n1, n2, n3, ...]
     */
    public static int[] calculateSizes(String stepStr) {
        String[] strArr = stepStr.split("-");
        String[] strArr2 = strArr[1].split("#");
        int start = Integer.valueOf(strArr[0]);
        int end = Integer.valueOf(strArr2[0]);
        int step = Integer.valueOf(strArr2[1]);

        int n = ((int) (end-start)/step) + 1;

        int[] sizes = new int[n];
        for(int i = 0; i < sizes.length; i++) {
            sizes[i] = start+step*i;
        }
        return sizes;
    }
}
