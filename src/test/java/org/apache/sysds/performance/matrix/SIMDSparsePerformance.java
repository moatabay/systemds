package org.apache.sysds.performance.matrix;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Power;
import org.apache.sysds.runtime.matrix.data.*;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

public class SIMDSparsePerformance {

	private static final String BASE_PATH = "vector_api_sparse_test/";
	private static final double EPSILON = 1E-9;

	private static long t1, t2, t3;
	private static double avg1, avg2, avg3, improvement;
	private static boolean resEqual1;

	public static void matrixMultTest(double[] sparsities1, double[] sparsities2, int rows, int cols1, int cols2,
                                      int k, int warmupRuns) {
        createBasePath();

        String outputPath = getOutputPathMultTest(sparsities1[0], sparsities2[0], rows, cols1, cols1, cols2, k);
        startWarmupMultTest(sparsities1[0], sparsities2[0], rows, cols1, cols2, k, warmupRuns);

        try(FileWriter writer = new FileWriter(outputPath)) {
            // Write CSV header
            writer.append("rows1,cols1,cols2,sparsity1,sparsity2,k,time_scalar,time_simd,improvement,correctness\n");

            for(double s1 : sparsities1) {
				for (double s2 : sparsities2) {
					MatrixBlock m1 = MatrixBlock.randOperations(rows, cols1, s1, -10, 10, "uniform", 7);
					MatrixBlock m2 = MatrixBlock.randOperations(cols1, cols2, s2, -10, 10, "uniform", 8);
					MatrixBlock retScalar = new MatrixBlock(rows, cols2, false);
					MatrixBlock retSIMD = new MatrixBlock(rows, cols2, false);

					avg1 = 0;
					avg2 = 0;
					avg3 = 0;

					// Measure execution time for the scalar multiplication.
					for (int i = 0; i < 30; i++) {
						t1 = System.nanoTime();
						LibMatrixMult.matrixMult(m1, m2, retScalar, k);
						avg1 += (System.nanoTime() - t1) / 1_000_000_000.0;
					}

					// Measure execution time for the SIMD multiplication.
					for (int i = 0; i < 30; i++) {
						t2 = System.nanoTime();
						LibMatrixMultSIMD.matrixMult(m1, m2, retSIMD, k); // SIMD
						avg2 += (System.nanoTime() - t2) / 1_000_000_000.0;
					}

					avg1 /= 30;
					avg2 /= 30;
					resEqual1 = compareResults(retSIMD, retScalar, rows, cols2);
					improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

					printStats(s1, s2, rows, cols1, cols1, cols2, k, false, false);

					// Write to csv
					writer.append(rows + "," + cols1 + "," + cols2 + "," + s1 + "," + s2 + "," + k + "," + avg1 + "," + avg2
							+ "," + improvement + "," + resEqual1 + "\n");
				}
			}
    	} catch(IOException e) {
			e.printStackTrace();
		} catch(Exception e) { // better for grep
			System.out.println("ERROR in Mult: ");
			e.printStackTrace();
		}
	}

	public static void matrixDivTest(double[] sparsities1, double[] sparsities2, double[] sparsities3, int rows, int cols,
		int mode, int k, int warmupRuns) {
		createBasePath();
		boolean sparseRet = sparsities3[0] < 0.4;

		String outputPath = getOutputPathDivTest(sparsities1[0], sparsities2[0], sparsities3[0], mode, k);
		startWarmupDivTest(sparsities1[0], sparsities2[0], sparseRet, rows, cols, mode, k, warmupRuns);

		try(FileWriter writer = new FileWriter(outputPath)) {
			// Write CSV header
			writer.append("rows1,cols1,sparsity1,sparsity2,k,time_scalar,time_simd,improvement,correctness\n");

			for(double s1 : sparsities1) {
				for(double s2 : sparsities2) {
					MatrixBlock m1 = MatrixBlock.randOperations(rows, cols, s1, -10, 10, "uniform", 7);
					MatrixBlock m2 = null;
					if(mode == 0) // matrix-matrix
						m2 = MatrixBlock.randOperations(rows, cols, s2, -10, 10, "uniform", 24);
					else if(mode == 1) // matrix-row vector
						m2 = MatrixBlock.randOperations(1, cols, s2, -10, 10, "uniform", 24);
					else if(mode == 2) // matrix-col vector
						m2 = MatrixBlock.randOperations(rows, 1, s2, -10, 10, "uniform", 24);

					MatrixBlock retScalar, retSIMD;
					retScalar = new MatrixBlock(rows, cols, sparseRet);
					retSIMD = new MatrixBlock(rows, cols, sparseRet);

					avg1 = 0;
					avg2 = 0;

					for(int i = 0; i < 10; i++) {
						retScalar.reset(rows, cols, sparseRet);
						t1 = System.nanoTime();
						LibMatrixBincell.bincellOp(m1, m2, retScalar, new BinaryOperator(Divide.getDivideFnObject()),
							k);
						avg1 += (System.nanoTime() - t1) / 1_000_000_000.0;
					}

					for(int i = 0; i < 10; i++) {
						retSIMD.reset(rows, cols, sparseRet);
						t2 = System.nanoTime();
						LibMatrixBincellSIMD.bincellOp(m1, m2, retSIMD, new BinaryOperator(Divide.getDivideFnObject()),
							k);
						avg2 += (System.nanoTime() - t2) / 1_000_000_000.0;
					}

					avg1 /= 10;
					avg2 /= 10;
					resEqual1 = compareResults(retScalar, retSIMD, rows, cols);
					improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

					printStats(s1, s2, rows, cols, rows, cols, k, false, false);

					// Write to csv
					writer.append(rows + "," + cols + "," + s1 + "," + s2 + "," + k + "," + avg1 + "," + avg2 + "," + improvement + ","
						+ resEqual1 + "\n");
				}
			}
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		catch(Exception e) { // better for grep
			System.out.println("ERROR in Div: ");
			e.printStackTrace();
		}
	}

	public static void matrixPowerTest(double[] sparsities, int rows, int cols, double exponent, int k,
		int warmupRuns) {
		createBasePath();

		String outputPath = getOutputPathPowerTest(exponent, k);
		startWarmupPowerTest(sparsities[0], rows, cols, exponent, k, warmupRuns);

		try(FileWriter writer = new FileWriter(outputPath)) {
			// Write CSV header
			writer.append("rows,cols,sparsity,k,time_scalar,time_simd,improvement,correctness\n");

			for(double sparsity : sparsities) {
				MatrixBlock m1 = MatrixBlock.randOperations(rows, cols, sparsity, -10, 10, "uniform", 7);
				MatrixBlock retScalar = new MatrixBlock(rows, cols, false);
				MatrixBlock retSIMD = new MatrixBlock(rows, cols, false);
				RightScalarOperator powerOpK = new RightScalarOperator(Power.getPowerFnObject(), exponent, k);

				avg1 = 0;
				avg2 = 0;

				for(int i = 0; i < 20; i++) {
					retScalar.reset(rows, cols, 0);
					t1 = System.nanoTime();
					retScalar = m1.scalarOperations(powerOpK, new MatrixBlock());
					avg1 += (System.nanoTime() - t1) / 1_000_000_000.0;
				}

				for(int i = 0; i < 20; i++) {
					retSIMD.reset(rows, cols, 0);
					t2 = System.nanoTime();
					retSIMD = m1.scalarOperationsSIMD(powerOpK, new MatrixBlock());
					avg2 += (System.nanoTime() - t2) / 1_000_000_000.0;
				}

				avg1 /= 20;
				avg2 /= 20;
				resEqual1 = compareResults(retScalar, retSIMD, rows, cols);
				improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

				printStats(sparsity, 0.0, rows, cols, 0, 0, k, false, true);

				// Write to csv
				writer.append(rows + "," + cols + "," + sparsity + "," + k + "," + avg1 + "," + avg2 + "," + improvement + "," + resEqual1 + "\n");
			}
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		catch(Exception e) { // better for grep
			System.out.println("ERROR in Power: ");
			e.printStackTrace();
		}
	}

	public static void matrixExpTest(double[] sparsities, int rows, int cols, int k, int warmupRuns) {
		createBasePath();

		String outputPath = getOutputPathExpTest(k);
		startWarmupExpTest(sparsities[0], rows, cols, k, warmupRuns);

		try(FileWriter writer = new FileWriter(outputPath)) {
			// Write CSV header
			writer.append("rows,cols,sparsity,k,time_scalar,time_simd,time_scalar2,improvement,correctness\n");

			for(double sparsity : sparsities) {
				MatrixBlock m1 = MatrixBlock.randOperations(rows, cols, sparsity, -10, 10, "uniform", 7);
				MatrixBlock retScalar = new MatrixBlock(rows, cols, false);
				MatrixBlock retScalar2 = new MatrixBlock(rows, cols, false);
				MatrixBlock retSIMD = new MatrixBlock(rows, cols, false);

				UnaryOperator expOperator = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.EXP), k);
				UnaryOperator exp2Operator = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.EXP_2),
					k);

				avg1 = 0;
				avg2 = 0;
				avg3 = 0;

				for(int i = 0; i < 10; i++) {
					retScalar.reset(rows, cols, 0);
					t1 = System.nanoTime();
					retScalar = m1.unaryOperations(expOperator, new MatrixBlock());
					avg1 += (System.nanoTime() - t1) / 1_000_000_000.0;
				}

				for(int i = 0; i < 10; i++) {
					retSIMD.reset(rows, cols, 0);
					t2 = System.nanoTime();
					retSIMD = m1.unaryOperationsSIMD(expOperator, new MatrixBlock());
					avg2 += (System.nanoTime() - t2) / 1_000_000_000.0;
				}

				for(int i = 0; i < 10; i++) {
					retScalar2.reset(rows, cols, 0);
					t3 = System.nanoTime();
					retScalar2 = m1.unaryOperations(exp2Operator, new MatrixBlock());
					avg3 += (System.nanoTime() - t3) / 1_000_000_000.0;
				}

				avg1 /= 10;
				avg2 /= 10;
				avg3 /= 10;
				resEqual1 = compareResults(retScalar, retSIMD, rows, cols);
				improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

				printStats(sparsity, 0.0, rows, cols, 0, 0, k, false, true);

				// Write to csv
				writer.append(rows + "," + cols + "," + sparsity + "," + k + "," + avg1 + "," + avg2 + "," + avg3 + "," + improvement + ","
					+ resEqual1 + "\n");
			}
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		catch(Exception e) { // better for grep
			System.out.println("ERROR in Exp: ");
			e.printStackTrace();
		}

	}

	private static void startWarmupMultTest(double sparsity1, double sparsity2, int rows, int cols1, int cols2, int k,
		int warmupRuns) {
		MatrixBlock m1 = MatrixBlock.randOperations(rows, cols1, sparsity1, -10, 10, "uniform", 7);
		MatrixBlock m2 = MatrixBlock.randOperations(cols1, cols2, sparsity2, -10, 10, "uniform", 8);

		for(int i = 0; i < warmupRuns; i++) {
			LibMatrixMult.matrixMult(m1, m2, k);
			LibMatrixMultSIMD.matrixMult(m1, m2, k);
		}
	}

	private static void startWarmupDivTest(double sparsity1, double sparsity2, boolean sparseRet, int rows1, int cols1,
		int mode, int k, int warmupRuns) {
		MatrixBlock m1 = MatrixBlock.randOperations(rows1, cols1, sparsity1, -10, 10, "uniform", 7);
		MatrixBlock m2 = null;
		if(mode == 0) // matrix-matrix
			m2 = MatrixBlock.randOperations(rows1, cols1, sparsity2, -10, 10, "uniform", 24);
		else if(mode == 1) // matrix-row vector
			m2 = MatrixBlock.randOperations(1, cols1, sparsity2, -10, 10, "uniform", 24);
		else if(mode == 2)
			m2 = MatrixBlock.randOperations(rows1, 1, sparsity2, -10, 10, "uniform", 24);
		MatrixBlock ret = new MatrixBlock(rows1, cols1, sparseRet);

		for(int i = 0; i < warmupRuns; i++) {
			LibMatrixBincell.bincellOp(m1, m2, ret, new BinaryOperator(Divide.getDivideFnObject(), k));
			LibMatrixBincellSIMD.bincellOp(m1, m2, ret, new BinaryOperator(Divide.getDivideFnObject(), k));
			ret.reset(rows1, cols1, sparseRet);
		}
	}

	private static void startWarmupPowerTest(double sparsity, int rows, int cols, double exponent, int k,
		int warmupRuns) {
		MatrixBlock m1 = MatrixBlock.randOperations(rows, cols, sparsity, 0, 10, "uniform", 7);
		RightScalarOperator powerOpK = new RightScalarOperator(Power.getPowerFnObject(), exponent, k);

		for(int i = 0; i < warmupRuns; i++) {
			m1.scalarOperations(powerOpK, new MatrixBlock());
			m1.scalarOperationsSIMD(powerOpK, new MatrixBlock());
		}
	}

	private static void startWarmupExpTest(double sparsity, int rows, int cols, int k, int warmupRuns) {
		MatrixBlock m1 = MatrixBlock.randOperations(rows, cols, sparsity, -10, 10, "uniform", 7);
		UnaryOperator expOperator = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.EXP), k);
		UnaryOperator exp2Operator = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.EXP_2), k);

		for(int i = 0; i < warmupRuns; i++) {
			m1.unaryOperations(expOperator, new MatrixBlock());
			m1.unaryOperations(exp2Operator, new MatrixBlock());
			m1.unaryOperationsSIMD(expOperator, new MatrixBlock());
		}
	}

	private static String getOutputPathMultTest(double sparsity1, double sparsity2, int rows1, int cols1, int rows2,
		int cols2, int k) {
		String prefix = "INVALID_";
		if(sparsity1 < 0.4 && sparsity2 >= 0.4) { // SPARSE DENSE
			if(rows1 > 17 && cols1 > 1 && cols2 > 1) {
				prefix = "sdmmmult_";
			}
			if(rows2 <= 2 * 1024 && cols2 == 1) {
				prefix = "sdmvmult_";
			}
		}
		else if(sparsity1 >= 0.4 && sparsity2 < 0.4) { // DENSE SPARSE
			prefix = "dsmmmult_";
		}
		return BASE_PATH + prefix + getTimeStamp() + "_s1=" + sparsity1 + "_s2=" + sparsity2 + "_k=" + k + ".csv";
	}

	private static String getOutputPathDivTest(double sparsity1, double sparsity2, double sparsity3, int mode, int k) {
		String prefix = "INVALID_";

		if(sparsity1 < 0.4 && sparsity2 >= 0.4) {
			// Check of sparsity3 is necessary because of potential invocation of safeBinaryMVSparseDenseRow
			if(sparsity3 >= 0.4 && mode == 0)
				prefix = "sdmmdiv_skip_";
			else if(sparsity3 < 0.4 && mode == 1)
				prefix = "sdmvdiv_row_";
			else if(sparsity3 >= 0.4 && mode == 2)
				prefix = "sdmvdiv_col_";
		}
		return BASE_PATH + prefix + getTimeStamp() + "_s1=" + sparsity1 + "_s2=" + sparsity2 + "_s3=" + sparsity3
			+ "_k=" + k + ".csv";
	}

	private static String getOutputPathPowerTest(double exponent, int k) {
		String prefix = "sparse_pow_";
		return BASE_PATH + prefix + getTimeStamp() + "_exponent=" + exponent + "_k=" + k + ".csv";
	}

	private static String getOutputPathExpTest(int k) {
		String prefix = "sparse_exp_";
		return BASE_PATH + prefix + getTimeStamp() + "_k=" + k + ".csv";
	}

	private static String getTimeStamp() {
		LocalDateTime now = LocalDateTime.now();
		DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss");
		return now.format(formatter);
	}

	private static void printStats(double sparsity1, double sparsity2, int rows1, int cols1, int rows2, int cols2,
		int k, boolean mklUsed, boolean singleMatrixOp) {
		String statImprovement = "Improvement scalar -> SIMD: " + improvement + "%";
		String statAverages = mklUsed ? "Averages - Scalar: " + avg1 + "; SIMD: " + avg2 + "; MKL: "
			+ avg3 : "Averages - Scalar: " + avg1 + "; SIMD: " + avg2;
		String statThreadsAndSparsities = "Threads: " + k
			+ (singleMatrixOp ? "; Sparsity: " + sparsity1 : "; Sparsity1: " + sparsity1 + "; Sparsity2: " + sparsity2);
		String statMatrices = singleMatrixOp ? rows1 + "x" + cols1 : "LHS: " + rows1 + "x" + cols1 + "; RHS: " + rows2
			+ "x" + cols2;
		String statResultsEqual = "Results are equal: " + resEqual1;

		System.out.println(statImprovement);
		System.out.println(statAverages);
		System.out.println(statThreadsAndSparsities);
		System.out.println(statMatrices);
		System.out.println(statResultsEqual);
		System.out.println("--------------------------------------------");
	}

	private static boolean compareResults(MatrixBlock mb1, MatrixBlock mb2, int rows, int cols) {
		if(mb1.getNonZeros() != mb2.getNonZeros()) {
			System.out.println("non-zeroes: " + mb1.getNonZeros() + " != " + mb2.getNonZeros());
			return false;
		}

		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				if(Math.abs(mb1.get(i, j) - mb2.get(i, j)) > EPSILON) {
					System.out.println("i=" + i + " j=" + j + ":" + mb1.get(i, j) + " is not equals " + mb2.get(i, j));
					return false;
				}
			}
		}
		return true;
	}

	private static void createBasePath() {
		try {
			Files.createDirectories(Paths.get(BASE_PATH));
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
	}
}
