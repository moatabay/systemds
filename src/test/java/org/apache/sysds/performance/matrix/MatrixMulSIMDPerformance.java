package org.apache.sysds.performance.matrix;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Power;
import org.apache.sysds.runtime.matrix.data.*;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

import java.io.File;
import java.io.FileNotFoundException;
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

    //TODO: Add multithreading to other tests

    public static void matrixMultTest(double sparsity1, double sparsity2, String rows1, String cols1, String cols2, int k, int warmupRuns, String dmlPath) {
        String outputPath = getOutputPathMultTest(sparsity1, sparsity2);

        try {
            Files.createDirectories(Paths.get(BASE_PATH));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        DMLConfig dmlConfig;

        try {
            dmlConfig = new DMLConfig(String.valueOf(new File(dmlPath)));
            ConfigurationManager.setGlobalConfig(dmlConfig);
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }

        int[] rowArr = calculateSizes(rows1);
        int[] col1Arr = calculateSizes(cols1);
        int[] col2Arr = calculateSizes(cols2);

        startWarmupMultTest(sparsity1, sparsity2, rowArr[0], col1Arr[0], col2Arr[0], k, warmupRuns);

        try (FileWriter writer = new FileWriter(outputPath)) {
            // Write CSV header
            writer.append("rows1,cols1,cols2,k,time_scalar,time_simd,time_mkl,improvement\n");

            for(int row : rowArr) {
                for (int col1 : col1Arr) {
                    for (int col2 : col2Arr) {
                        // Generate two random dense matrices
                        MatrixBlock mbA = MatrixBlock.randOperations(row, col1, sparsity1, 0, 1, "uniform", 7);
                        MatrixBlock mbB = MatrixBlock.randOperations(col1, col2, sparsity2, 0, 1, "uniform", 8);
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
                        resEqual1 = compareResults(resultA, resultB, row, col2);
                        resEqual2 = compareResults(resultA, ret, row, col2);
                        improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

                        printStats(row, col1, col1, col2, k, true, false);

                        // Write to csv
                        writer.append(  row + "," +
                                        col1 + "," +
                                        col2 + "," +
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

    public static void matrixDiffTest(double sparsity1, double sparsity2, String rows, String cols, int k, int warmupRuns) {
        String outputPath = getOutputPathDiffTest(sparsity1, sparsity2);

        try {
            Files.createDirectories(Paths.get(BASE_PATH));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        int[] rowArr = calculateSizes(rows);
        int[] colArr = calculateSizes(cols);

        startWarmupDiffTest(sparsity1, sparsity2, rowArr[0], colArr[0], k, warmupRuns);

        try (FileWriter writer = new FileWriter(outputPath)) {
            // Write CSV header
            writer.append("rows,cols,k,time_scalar,time_simd,improvement\n");

            for(int row : rowArr) {
                for(int col : colArr) {
                    MatrixBlock mbA = MatrixBlock.randOperations(row, col, sparsity1, 0, 1, "uniform", 7);
                    MatrixBlock mbB = MatrixBlock.randOperations(row, col, sparsity2, 0, 1, "uniform", 8);
                    MatrixBlock retScalar = new MatrixBlock(row, col, false);
                    MatrixBlock retSIMD = MatrixBlock.randOperations(row, col, sparsity1, 0, 1, "uniform", 9);

                    avg1 = 0;
                    avg2 = 0;

                    for(int i = 0; i < 10; i++) {
                        retScalar.reset(row, col, 0); // prevent "cumulative" subtraction
                        t1 = System.nanoTime();
                        LibMatrixBincell.bincellOp(mbA, mbB, retScalar, new BinaryOperator(Minus.getMinusFnObject()), 16);
                        avg1 += (System.nanoTime() - t1) / 1000000;
                    }

                    for(int i = 0; i < 10; i++) {
                        t2 = System.nanoTime();
                        LibMatrixEWOP.diff(mbA, mbB, retSIMD);
                        avg2 += (System.nanoTime() - t2) / 1000000;
                    }

                    avg1 /= 10;
                    avg2 /= 10;
                    resEqual1 = compareResults(retScalar, retSIMD, row, col);
                    improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

                    printStats(row, col, row, col, k, false, false);

                    // Write to csv
                    writer.append(  row + "," +
                            col + "," +
                            k + "," +
                            avg1 + "," +
                            avg2 + "," +
                            improvement + "\n");
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void matrixPowerTest(double sparsity, String rows, String cols, double exponent, int k, int warmupRuns) {
        String outputPath = getOutputPathPowerTest(sparsity, exponent);

        try {
            Files.createDirectories(Paths.get(BASE_PATH));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        int[] rowArr = calculateSizes(rows);
        int[] colArr = calculateSizes(cols);

        startWarmupPowerTest(sparsity, rowArr[0], colArr[0], exponent, k, warmupRuns);

        try (FileWriter writer = new FileWriter(outputPath)) {
            // Write CSV header
            writer.append("rows,cols,k,time_scalar,time_simd,improvement\n");

            for(int row : rowArr) {
                for(int col : colArr) {
                    MatrixBlock mbA = MatrixBlock.randOperations(row, col, sparsity, 0, 1, "uniform", 7);
                    MatrixBlock retScalar = new MatrixBlock(row, col, false);
                    MatrixBlock retSIMD = MatrixBlock.randOperations(row, col, sparsity, 0, 1, "uniform", 9);
                    RightScalarOperator powerOpK = new RightScalarOperator(Power.getPowerFnObject(), exponent);

                    avg1 = 0;
                    avg2 = 0;

                    for(int i = 0; i < 10; i++) {
                        retScalar.reset(row, col, 0);
                        t1 = System.nanoTime();
                        retScalar = mbA.scalarOperations(powerOpK, new MatrixBlock());
                        avg1 += (System.nanoTime() - t1) / 1000000;
                    }

                    for(int i = 0; i < 10; i++) {
                        t2 = System.nanoTime();
                        LibMatrixEWOP.power(mbA, retSIMD, exponent);
                        avg2 += (System.nanoTime() - t2) / 1000000;
                    }

                    avg1 /= 10;
                    avg2 /= 10;
                    resEqual1 = compareResults(retScalar, retSIMD, row, col);
                    improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

                    printStats(row, col, 0, 0, k, false, true);

                    // Write to csv
                    writer.append(  row + "," +
                            col + "," +
                            k + "," +
                            avg1 + "," +
                            avg2 + "," +
                            improvement + "\n");
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    public static void matrixExpTest(double sparsity, String rows, String cols, int k, int warmupRuns) {
        String outputPath = getOutputPathExpTest(sparsity);

        try {
            Files.createDirectories(Paths.get(BASE_PATH));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        int[] rowArr = calculateSizes(rows);
        int[] colArr = calculateSizes(cols);

        startWarmupExpTest(sparsity, rowArr[0], colArr[0], k, warmupRuns);

        try (FileWriter writer = new FileWriter(outputPath)) {
            // Write CSV header
            writer.append("rows,cols,k,time_scalar,time_simd,improvement\n");

            for(int row : rowArr) {
                for(int col : colArr) {
                    MatrixBlock mbA = MatrixBlock.randOperations(row, col, sparsity, 0, 1, "uniform", 7);
                    MatrixBlock retScalar = new MatrixBlock(row, col, false);
                    MatrixBlock retSIMD = MatrixBlock.randOperations(row, col, sparsity, 0, 1, "uniform", 9);

                    UnaryOperator expOperator = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.EXP));

                    avg1 = 0;
                    avg2 = 0;

                    for(int i = 0; i < 10; i++) {
                        retScalar.reset(row, col, 0);
                        t1 = System.nanoTime();
                        retScalar = mbA.unaryOperations(expOperator, new MatrixBlock());
                        avg1 += (System.nanoTime() - t1) / 1000000;
                    }

                    for(int i = 0; i < 10; i++) {
                        t2 = System.nanoTime();
                        LibMatrixEWOP.exp(mbA, retSIMD);
                        avg2 += (System.nanoTime() - t2) / 1000000;
                    }

                    avg1 /= 10;
                    avg2 /= 10;
                    resEqual1 = compareResults(retScalar, retSIMD, row, col);
                    improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

                    printStats(row, col, 0, 0, k, false, true);

                    // Write to csv
                    writer.append(  row + "," +
                            col + "," +
                            k + "," +
                            avg1 + "," +
                            avg2 + "," +
                            improvement + "\n");
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    private static void startWarmupMultTest(double sparsity1, double sparsity2, int rows, int cols1, int cols2, int k, int warmupRuns) {
        MatrixBlock warmUpA = MatrixBlock.randOperations(rows, cols1, sparsity1, 0, 1, "uniform", 7);
        MatrixBlock warmUpB = MatrixBlock.randOperations(cols1, cols2, sparsity2, 0, 1, "uniform", 8);
        MatrixBlock warmUpRet = new MatrixBlock(rows, cols2, false);

        for(int i = 0; i < warmupRuns; i++) {
            LibMatrixMult2.matrixMult(warmUpA, warmUpB, k);
            LibMatrixMult.matrixMult(warmUpA, warmUpB, k);
            LibMatrixNative.matrixMult(warmUpA, warmUpB, warmUpRet, k);
        }
    }

    private static void startWarmupDiffTest(double sparsity1, double sparsity2, int rows, int cols, int k, int warmupRuns) {
        MatrixBlock mbA = MatrixBlock.randOperations(rows, cols, sparsity1, 0, 1, "uniform", 7);
        MatrixBlock mbB = MatrixBlock.randOperations(rows, cols, sparsity2, 0, 1, "uniform", 8);
        MatrixBlock warmUpRet = new MatrixBlock(rows, cols, false);

        for(int i = 0; i < warmupRuns; i++) {
			LibMatrixBincell.bincellOp(mbA, mbB, warmUpRet, new BinaryOperator(Minus.getMinusFnObject()));
			LibMatrixEWOP.diff(mbA, mbB, warmUpRet);
        }
    }

    private static void startWarmupPowerTest(double sparsity, int rows, int cols, double exponent, int k, int warmupRuns) {
        MatrixBlock mbA = MatrixBlock.randOperations(rows, cols, sparsity, 0, 1, "uniform", 7);
        MatrixBlock warmUpRet;
        RightScalarOperator powerOpK = new RightScalarOperator(Power.getPowerFnObject(), exponent);

        for(int i = 0; i < warmupRuns; i++) {
            warmUpRet = mbA.scalarOperations(powerOpK, new MatrixBlock());
            LibMatrixEWOP.power(mbA, warmUpRet, exponent);
        }
    }

    private static void startWarmupExpTest(double sparsity, int rows, int cols, int k, int warmupRuns) {
        MatrixBlock mbA = MatrixBlock.randOperations(rows, cols, sparsity, 0, 1, "uniform", 7);
        MatrixBlock warmUpRet;
        UnaryOperator expOperator = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.EXP));

        for(int i = 0; i < warmupRuns; i++) {
            warmUpRet = mbA.unaryOperations(expOperator, new MatrixBlock());
            LibMatrixEWOP.exp(mbA, warmUpRet);
        }
    }

    public static String getOutputPathMultTest(double sparsity1, double sparsity2) {
        return BASE_PATH + "perfMult_" + getTimeStamp() + "_s1=" + sparsity1 + "_s2=" + sparsity2 + ".csv";
    }

    public static String getOutputPathDiffTest(double sparsity1, double sparsity2) {
        return BASE_PATH + "perfDiff_" + getTimeStamp() + "_s1=" + sparsity1 + "_s2=" + sparsity2 + ".csv";
    }

    public static String getOutputPathPowerTest(double sparsity1, double exponent) {
        return BASE_PATH + "perfPower_" + getTimeStamp() + "_s1=" + sparsity1 + "_exponent=" + exponent + ".csv";
    }

    public static String getTimeStamp() {
        LocalDateTime now = LocalDateTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss");
        return now.format(formatter);
    }

    public static String getOutputPathExpTest(double sparsity1) {
        LocalDateTime now = LocalDateTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss");
        String formattedNow = now.format(formatter);

        return BASE_PATH + "perfExp_" + formattedNow + "_s1=" + sparsity1 + ".csv";
    }

    public static void printStats(int rows1, int cols1, int rows2, int cols2, int k, boolean mklUsed, boolean singleMatrixOp) {
        String statImprovement = "Improvement scalar -> SIMD: " + improvement + "%";
        String statAverages = mklUsed ? "Averages - Scalar: " + avg1 + "; SIMD: " + avg2 + "; MKL: " + avg3 :
                                    "Averages - Scalar: " + avg1 + "; SIMD: " + avg2;
        String statMatrices = singleMatrixOp ? rows1 + "x" + cols1 : "LHS: " + rows1 + "x" + cols1 + "; RHS: " + rows2 + "x" + cols2;
        String statResultsEqual = "Results are equal: " + ((mklUsed) ? (resEqual1 && resEqual2) : resEqual1);

        System.out.println(statImprovement);
        System.out.println(statAverages);
        System.out.println(statMatrices);
        System.out.println(statResultsEqual);
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
