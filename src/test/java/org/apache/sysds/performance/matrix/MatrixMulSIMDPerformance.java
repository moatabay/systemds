package org.apache.sysds.performance.matrix;

import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Power;
import org.apache.sysds.runtime.matrix.data.*;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

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
    
    public static void matrixMultTest(double sparsity1, double sparsity2, String rows1, String cols1, String cols2, int k, int warmupRuns) {
        String outputPath = getOutputPathMultTest(sparsity1, sparsity2);

        try {
            Files.createDirectories(Paths.get(BASE_PATH));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        int[] rows = calculateSizes(rows1);
        int[] columns1 = calculateSizes(cols1);
        int[] columns2 = calculateSizes(cols2);

        startWarmupMultTest(sparsity1, sparsity2, rows[0], columns1[0], columns2[0], k, warmupRuns);

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

            for(int rl : rowArr) {
                for(int cl : colArr) {
                    MatrixBlock mbA = MatrixBlock.randOperations(rowArr[0], colArr[0], sparsity1, 0, 1, "uniform", 7);
                    MatrixBlock mbB = MatrixBlock.randOperations(rowArr[0], colArr[0], sparsity2, 0, 1, "uniform", 8);
                    MatrixBlock retScalar = new MatrixBlock(rowArr[0], colArr[0], false);
                    MatrixBlock retSIMD = MatrixBlock.randOperations(rowArr[0], colArr[0], sparsity1, 0, 1, "uniform", 9);

                    avg1 = 0;
                    avg2 = 0;

                    for(int i = 0; i < 10; i++) {
                        t1 = System.nanoTime();
                        LibMatrixBincell.bincellOp(mbA, mbB, retScalar, new BinaryOperator(Minus.getMinusFnObject()));
                        avg1 += (System.nanoTime() - t1) / 1000000;
                    }

                    for(int i = 0; i < 10; i++) {
                        t2 = System.nanoTime();
                        LibMatrixEWOP.diff(mbA, mbB, retSIMD);
                        avg2 += (System.nanoTime() - t2) / 1000000;
                    }

                    avg1 /= 10;
                    avg2 /= 10;
                    resEqual1 = compareResults(retScalar, retSIMD, rl, cl);
                    improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

                    printStats(rl, cl, 0, k);

                    // Write to csv
                    writer.append(  rl + "," +
                            cl + "," +
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

        startWarmupPowerTest(sparsity, exponent, rowArr[0], colArr[0], k, warmupRuns);

        try (FileWriter writer = new FileWriter(outputPath)) {
            // Write CSV header
            writer.append("rows,cols,k,time_scalar,time_simd,improvement\n");

            for(int rl : rowArr) {
                for(int cl : colArr) {
                    MatrixBlock mbA = MatrixBlock.randOperations(rl, cl, sparsity, 0, 1, "uniform", 7);
                    MatrixBlock retScalar = new MatrixBlock(rl, cl, false);
                    MatrixBlock retSIMD = MatrixBlock.randOperations(rl, cl, sparsity, 0, 1, "uniform", 9);
                    RightScalarOperator powerOpK = new RightScalarOperator(Power.getPowerFnObject(), exponent);

                    avg1 = 0;
                    avg2 = 0;

                    for(int i = 0; i < 10; i++) {
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
                    resEqual1 = compareResults(retScalar, retSIMD, rl, cl);
                    improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

                    printStats(rl, cl, 0, k);

                    // Write to csv
                    writer.append(  rl + "," +
                            cl + "," +
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

            for(int rl : rowArr) {
                for(int cl : colArr) {
                    MatrixBlock mbA = MatrixBlock.randOperations(rl, cl, sparsity, 0, 1, "uniform", 7);
                    MatrixBlock retScalar = new MatrixBlock(rl, cl, false);
                    MatrixBlock retSIMD = MatrixBlock.randOperations(rl, cl, sparsity, 0, 1, "uniform", 9);

                    UnaryOperator expOperator = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.EXP));

                    avg1 = 0;
                    avg2 = 0;

                    for(int i = 0; i < 10; i++) {
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
                    resEqual1 = compareResults(retScalar, retSIMD, rl, cl);
                    improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

                    printStats(rl, cl, 0, k);

                    // Write to csv
                    writer.append(  rl + "," +
                            cl + "," +
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

    private static void startWarmupMultTest(double sparsity1, double sparsity2, int rl, int cl1, int cl2, int k, int warmupRuns) {
        MatrixBlock warmUpA = MatrixBlock.randOperations(rl, cl1, sparsity1, 0, 1, "uniform", 7);
        MatrixBlock warmUpB = MatrixBlock.randOperations(cl1, cl2, sparsity2, 0, 1, "uniform", 8);
        MatrixBlock warmUpRet = new MatrixBlock(rl, cl2, false);

        for(int i = 0; i < warmupRuns; i++) {
            LibMatrixMult2.matrixMult(warmUpA, warmUpB, k);
            LibMatrixMult.matrixMult(warmUpA, warmUpB, k);
            LibMatrixNative.matrixMult(warmUpA, warmUpB, warmUpRet, k);
        }
    }

    private static void startWarmupDiffTest(double sparsity1, double sparsity2, int rl, int cl, int k, int warmupRuns) {
        MatrixBlock mbA = MatrixBlock.randOperations(rl, cl, sparsity1, 0, 1, "uniform", 7);
        MatrixBlock mbB = MatrixBlock.randOperations(rl, cl, sparsity2, 0, 1, "uniform", 8);
        MatrixBlock warmUpRet = new MatrixBlock(rl, cl, false);

        for(int i = 0; i < warmupRuns; i++) {
			LibMatrixBincell.bincellOp(mbA, mbB, warmUpRet, new BinaryOperator(Minus.getMinusFnObject()));
			LibMatrixEWOP.diff(mbA, mbB, warmUpRet);
        }
    }

    private static void startWarmupPowerTest(double sparsity, double exponent, int rl, int cl, int k, int warmupRuns) {
        MatrixBlock mbA = MatrixBlock.randOperations(rl, cl, sparsity, 0, 1, "uniform", 7);
        MatrixBlock warmUpRet;
        RightScalarOperator powerOpK = new RightScalarOperator(Power.getPowerFnObject(), exponent);

        for(int i = 0; i < warmupRuns; i++) {
            warmUpRet = mbA.scalarOperations(powerOpK, new MatrixBlock());
            LibMatrixEWOP.power(mbA, warmUpRet, exponent);
        }
    }

    private static void startWarmupExpTest(double sparsity, int rl, int cl, int k, int warmupRuns) {
        MatrixBlock mbA = MatrixBlock.randOperations(rl, cl, sparsity, 0, 1, "uniform", 7);
        MatrixBlock warmUpRet;
        UnaryOperator expOperator = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.EXP));

        for(int i = 0; i < warmupRuns; i++) {
            warmUpRet = mbA.unaryOperations(expOperator, new MatrixBlock());
            LibMatrixEWOP.exp(mbA, warmUpRet);
        }
    }

    public static String getOutputPathMultTest(double sparsity1, double sparsity2) {
        LocalDateTime now = LocalDateTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss");
        String formattedNow = now.format(formatter);

        return BASE_PATH + "perfMult_" + formattedNow + "_s1=" + sparsity1 + "_s2=" + sparsity2 + ".csv";
    }

    public static String getOutputPathDiffTest(double sparsity1, double sparsity2) {
        LocalDateTime now = LocalDateTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss");
        String formattedNow = now.format(formatter);

        return BASE_PATH + "perfDiff_" + formattedNow + "_s1=" + sparsity1 + "_s2=" + sparsity2 + ".csv";
    }

    public static String getOutputPathPowerTest(double sparsity1, double exponent) {
        LocalDateTime now = LocalDateTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss");
        String formattedNow = now.format(formatter);

        return BASE_PATH + "perfPower_" + formattedNow + "_s1=" + sparsity1 + "_exponent=" + exponent + ".csv";
    }

    public static String getOutputPathExpTest(double sparsity1) {
        LocalDateTime now = LocalDateTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss");
        String formattedNow = now.format(formatter);

        return BASE_PATH + "perfExp_" + formattedNow + "_s1=" + sparsity1 + ".csv";
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
