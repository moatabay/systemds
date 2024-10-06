package org.apache.sysds.performance.matrix;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.matrix.data.*;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import java.nio.file.Paths;
import java.nio.file.Files;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class MatrixMulSIMDPerformance {

    private static final String BASE_PATH = "vector_api_test/";
    private static final double EPSILON = 1E-10;

    private static long t1, t2, t3;
    private static double avg1, avg2, avg3, improvement;
    private static MatrixBlock resultA = null, resultB = null;
    private static boolean resEqual1, resEqual2;
    private final static File TEST_CONF_FILE = new File("src/test/config/SystemDS-config.xml");

    public static void matrixMultTest(double sparsity1, double sparsity2, String rows1, String cols1, String cols2, int k, int warmupRuns) {
        String outputPath = getOutputPath(sparsity1, sparsity2);

        DMLConfig dmlconf = null;

        try {
            Files.createDirectories(Paths.get(BASE_PATH));
            dmlconf = DMLConfig.readConfigurationFile(TEST_CONF_FILE.getPath());

            ConfigurationManager.setGlobalConfig(dmlconf);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        int[] rows = calculateSizes(rows1);
        int[] columns1 = calculateSizes(cols1);
        int[] columns2 = calculateSizes(cols2);

        startWarmup(sparsity1, sparsity2, rows[0], columns1[0], columns2[0], k, warmupRuns);

        try (FileWriter writer = new FileWriter(outputPath)) {
            // Write CSV header
            writer.append("rows1,cols1,cols2,k,time_scalar,time_simd,time_mkl,improvement\n");

            for(int rl : rows) {
                for (int cl1 : columns1) {
                    for (int cl2 : columns2) {
                        // Generate two random dense matrices
                        MatrixBlock mbA = MatrixBlock.randOperations(rl, cl1, sparsity1, 0, 1, "uniform", 7);
                        MatrixBlock mbB = MatrixBlock.randOperations(cl1, cl2, sparsity2, 0, 1, "uniform", 8);
                        MatrixBlock ret = new MatrixBlock(mbA.getNumRows(), mbB.getNumColumns(), false);

                        avg1 = 0;
                        avg2 = 0;
                        avg3 = 0;

                        // Measure execution time for the scalar multiplication.
                        for(int i = 0; i < 10; i++) {
                            t1 = System.nanoTime();
                            resultA = LibMatrixMult2.matrixMult(mbA, mbB, k);

                            avg1 += (System.nanoTime() - t1) / 1000000;
                        }

                        // Measure execution time for the SIMD multiplication.
                        for(int i = 0; i < 10; i++) {
                            t2 = System.nanoTime();
                            resultB = LibMatrixMult.matrixMult(mbA, mbB, k); // SIMD

                            avg2 += (System.nanoTime() - t2) / 1000000;
                        }

                        // Measure execution time for the MKL multiplication.
                        for(int i = 0; i < 10; i++) {
                            t3 = System.nanoTime();
                            LibMatrixNative.matrixMult(mbA, mbB, ret, k);

                            avg3 += (System.nanoTime() - t3) / 1000000;
                        }

                        avg1 /= 10;
                        avg2 /= 10;
                        avg3 /= 10;
                        resEqual1 = compareResults(resultA, resultB, rl, cl2);
                        resEqual2 = compareResults(resultA, ret, rl, cl2);
                        improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

                        printStats(rl, cl1, cl2, k);

                        // Write to csv
                        writer.append(  rl + "," +
                                        cl1 + "," +
                                        cl2 + "," +
                                        k + "," +
                                        avg1 + "," +
                                        avg2 + "," +
                                        avg3 + "," +
                                        improvement + "\n");
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void startWarmup(double sparsity1, double sparsity2, int rl, int cl1, int cl2, int k, int warmupRuns) {
        MatrixBlock warmUpA = MatrixBlock.randOperations(rl, cl1, sparsity1, 0, 1, "uniform", 7);
        MatrixBlock warmUpB = MatrixBlock.randOperations(cl1, cl2, sparsity2, 0, 1, "uniform", 8);
        MatrixBlock warmUpRet = new MatrixBlock(rl, cl2, false);

        for(int i = 0; i < warmupRuns; i++) {
            LibMatrixMult.matrixMult(warmUpA, warmUpB, k);
            LibMatrixMult2.matrixMult(warmUpA, warmUpB, k);
            LibMatrixNative.matrixMult(warmUpA, warmUpB, warmUpRet, k);
        }
    }

    public static String getOutputPath(double sparsity1, double sparsity2) {
        LocalDateTime now = LocalDateTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss");
        String formattedNow = now.format(formatter);

        return BASE_PATH + "performance1_" + formattedNow + "_s1=" + sparsity1 + "_s2=" + sparsity2 + ".csv";
    }

    public static void printStats(int rows1, int cols1, int cols2, int k) {
        System.out.println("Improvement scalar -> SIMD: " + improvement + "%");
        System.out.println("Averages - Scalar: " + avg1 + "; SIMD: " + avg2 + "; MKL: " + avg3);
        System.out.println(rows1 + "x" + cols1 + " * " + cols1 + "x" + cols2 + "; threads = " + k);
        System.out.println("Results are equal: " + (resEqual1 && resEqual2));
        System.out.println("--------------------------------------------");
    }

    public static boolean compareResults(MatrixBlock mb1, MatrixBlock mb2, int rows, int cols) {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                if(Math.abs(mb1.get(i,j)-mb2.get(i,j)) > EPSILON) {
                    System.out.println(mb1.get(i,j) + " is not equals " + mb2.get(i,j));
                    return false;
                }
            }
        }
        return true;
    }

    /**
     *
     * @param stepStr
     * @return sizes-Array [n1, n2, n3, ...]
     */
    public static int[] calculateSizes(String stepStr) {
        String[] strArr = stepStr.split("-");

        if(strArr.length == 1) {
            // stepStr only contains 1 value
            return new int[]{Integer.parseInt(strArr[0])};
        }

        // Continue with split string
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
