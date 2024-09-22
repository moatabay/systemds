package org.apache.sysds.performance.matrix;

import org.apache.commons.math3.analysis.function.Exp;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Power;
import org.apache.sysds.runtime.functionobjects.Power2;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult2;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
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

    public static void squareMMSIMDTest(double sparsityMatrix1, double sparsityMatrix2, String kSteps, String nSteps, int warmUpIterations) {
        String outputPath = BASE_PATH + "performance1_" + getTimeStamp() + ".csv";
        long startTime1 = 0, endTime1 = 0, startTime2 = 0, endTime2 = 0;
        double avg1 = 0.0, avg2 = 0.0, improvement = 0.0;
        MatrixBlock resultA = null, resultB = null;
        boolean resEqual = false;

        try {
            Files.createDirectories(Paths.get(BASE_PATH));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        try (FileWriter writer = new FileWriter(outputPath)) {
            // Write CSV header
            writer.append("n,k,exec_time_no_simd,exec_time_simd,improvement\n");

            int[] kSizes = calculateSizes(kSteps);
            int[] nSizes = calculateSizes(nSteps);

            MatrixBlock warmUpA = MatrixBlock.randOperations(2000, 2000, sparsityMatrix1, 0, 1, "uniform", 7);
            MatrixBlock warmUpB = MatrixBlock.randOperations(2000, 2000, sparsityMatrix2, 0, 1, "uniform", 7);

            for(int i = 0; i < warmUpIterations; i++) {
                LibMatrixMult.matrixMult(warmUpA, warmUpB, kSizes[0]);
                LibMatrixMult2.matrixMult(warmUpA, warmUpB, kSizes[0]);
            }

            for (int k : kSizes) {
                for (int n : nSizes) {
                    // Generate two random dense matrices
                    MatrixBlock mbA = MatrixBlock.randOperations(n, n, sparsityMatrix1, 0, 1, "uniform", 7);
                    MatrixBlock mbB = MatrixBlock.randOperations(n, n, sparsityMatrix2, 0, 1, "uniform", 7);

                    // Measure the execution time of the matrix multiplication
                    avg1 = 0;
                    avg2 = 0;

                    for(int i = 0; i < 5; i++) {
                        LibMatrixMult2.matrixMult(mbA, mbB, k); // No SIMD
                    }

                    for(int i = 0; i < 10; i++) {
                        startTime1 = System.nanoTime();
                        resultA = LibMatrixMult2.matrixMult(mbA, mbB, k); // No SIMD
                        endTime1 = System.nanoTime();

                        avg1 += (endTime1 - startTime1) / 1000000;
                    }

                    for(int i = 0; i < 5; i++) {
                        LibMatrixMult.matrixMult(mbA, mbB, k); // SIMD
                    }

                    for(int i = 0; i < 10; i++) {
                        startTime2 = System.nanoTime();
                        resultB = LibMatrixMult.matrixMult(mbA, mbB, k); // SIMD
                        endTime2 = System.nanoTime();

                        avg2 += (endTime2 - startTime2) / 1000000;
                    }

                    resEqual = compareResults(resultA, resultB, n, n);
                    improvement = Math.round(((avg1-avg2)/avg1)*100.0 * 100.0) / 100.0;

                    System.out.println("Averages - NO-SIMD: " + avg1/10
                            + "; SIMD: " + avg2/10
                            + "; n=" + n
                            + "; k=" + k
                            + "; Improvement: " + improvement + "%");
                    System.out.println("Results are equal: " + resEqual);
                    System.out.println("------------------");
                    // Write to csv
                    writer.append(n + "," + k + "," + avg1/10 + "," + avg2/10 + "," + improvement + "\n");
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void MVSIMDTest(double sparsityMatrix1, double sparsityMatrix2, String kSteps, int rows, int cols, int warmUpIterations) {
        if(cols > 2*1024) {
            System.out.println("Illegal amount of columns. Must be <= 2*1024!");
            return;
        }

        String outputPath = BASE_PATH + "performance2_" + getTimeStamp() + ".csv";
        long startTime1 = 0, endTime1 = 0, startTime2 = 0, endTime2 = 0;
        double avg1 = 0.0, avg2 = 0.0, improvement = 0.0;
        MatrixBlock resultA = null, resultB = null;
        boolean resEqual = false;

        try {
            Files.createDirectories(Paths.get(BASE_PATH));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        try (FileWriter writer = new FileWriter(outputPath)) {
            // Write CSV header
            writer.append("rows,cols,k,exec_time_no_simd,exec_time_simd,improvement\n");

            int[] kSizes = calculateSizes(kSteps);

            MatrixBlock warmUpA = MatrixBlock.randOperations(300000, 2048, sparsityMatrix1, 0, 1, "uniform", 7);
            MatrixBlock warmUpB = MatrixBlock.randOperations(2048, 1, sparsityMatrix2, 0, 1, "uniform", 7);

            for(int i = 0; i < warmUpIterations; i++) {
                LibMatrixMult.matrixMult(warmUpA, warmUpB, kSizes[0]);
                LibMatrixMult2.matrixMult(warmUpA, warmUpB, kSizes[0]);
            }

            for (int k : kSizes) {
				// Generate two random dense matrices
				MatrixBlock mbA = MatrixBlock.randOperations(rows, cols, sparsityMatrix1, 0, 1, "uniform", 7);
				MatrixBlock mbB = MatrixBlock.randOperations(cols, 1, sparsityMatrix2, 0, 1, "uniform", 7);

				// Measure the execution time of the matrix multiplication
				avg1 = 0;
				avg2 = 0;

				for(int i = 0; i < 5; i++) {
					LibMatrixMult2.matrixMult(mbA, mbB, k); // No SIMD
				}

				for(int i = 0; i < 10; i++) {
					startTime1 = System.nanoTime();
					resultA = LibMatrixMult2.matrixMult(mbA, mbB, k); // No SIMD
					endTime1 = System.nanoTime();

					avg1 += (endTime1 - startTime1) / 1000000;
				}

				for(int i = 0; i < 5; i++) {
					LibMatrixMult.matrixMult(mbA, mbB, k); // SIMD
				}

				for(int i = 0; i < 10; i++) {
					startTime2 = System.nanoTime();
					resultB = LibMatrixMult.matrixMult(mbA, mbB, k); // SIMD
					endTime2 = System.nanoTime();

					avg2 += (endTime2 - startTime2) / 1000000;
				}

				resEqual = compareResults(resultA, resultB, resultA.getNumRows(), resultA.getNumColumns());
				improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

				System.out.println("Averages - NO-SIMD: " + avg1 / 10 + "; SIMD: " + avg2 / 10 + "; rows=" + rows + "; cols=" + cols + "; k=" + k
					+ "; Improvement: " + improvement + "%");
				System.out.println("Results are equal: " + resEqual);
				System.out.println("------------------");
				// Write to csv
				writer.append(rows + "," + cols + "," + k + "," + avg1 / 10 + "," + avg2 / 10 + "," + improvement + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void diffExpPowerTest() {
        MatrixBlock A = MatrixBlock.randOperations(2, 2, 1, 0, 1, "uniform", 7);
        MatrixBlock B = MatrixBlock.randOperations(2, 2, 1, 0, 1, "uniform", 8);
        MatrixBlock minusC = new MatrixBlock(A.getNumRows(), A.getNumColumns(), false);
        MatrixBlock powerC = new MatrixBlock(A.getNumRows(), A.getNumColumns(), false);
        MatrixBlock expC = new MatrixBlock(A.getNumRows(), A.getNumColumns(), false);

        LibMatrixBincell.bincellOp(A, B, minusC, new BinaryOperator(Minus.getMinusFnObject()));
        LibMatrixBincell.uncellOp(A, powerC, new UnaryOperator(Power2.getPower2FnObject()));
        System.out.println("A: " + A);
        System.out.println("B: " + B);
        System.out.println("C Minus: " + minusC);
        System.out.println("C Power: " + powerC);
        System.out.println("C Exp: " + expC);
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

    public static String getTimeStamp() {
        LocalDateTime now = LocalDateTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss");
        String formattedNow = now.format(formatter);
        return formattedNow;
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
}
