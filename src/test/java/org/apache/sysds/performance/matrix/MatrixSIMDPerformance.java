package org.apache.sysds.performance.matrix;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Multiply;
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

public class MatrixSIMDPerformance {

	private static final String BASE_PATH = "vector_api_test/";
	private static final double EPSILON = 1E-9;

	private static long t1, t2, t3;
	private static double avg1, avg2, avg3, improvement;
	private static boolean resEqual1, resEqual2;

	public static void matrixMultTest(double sparsity1, double sparsity2, String rows1, String cols1, String cols2,
		int k, int warmupRuns, String dmlPath) {

		try {
			Files.createDirectories(Paths.get(BASE_PATH));
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}

		DMLConfig dmlConfig;

		try {
			dmlConfig = new DMLConfig(String.valueOf(new File(dmlPath)));
			ConfigurationManager.setGlobalConfig(dmlConfig);
		}
		catch(FileNotFoundException e) {
			throw new RuntimeException(e);
		}

		int[] rowArr = calculateSizes(rows1);
		int[] col1Arr = calculateSizes(cols1);
		int[] col2Arr = calculateSizes(cols2);

		String outputPath = getOutputPathMultTest(sparsity1, sparsity2, rowArr[0], col1Arr[0], col1Arr[0], col2Arr[0], k);
		startWarmupMultTest(sparsity1, sparsity2, rowArr[0], col1Arr[0], col2Arr[0], k, warmupRuns);

		try(FileWriter writer = new FileWriter(outputPath)) {
			// Write CSV header
			writer.append("rows1,cols1,cols2,k,time_scalar,time_simd,time_mkl,improvement,correctness\n");

			for(int row : rowArr) { // Varying row sizes for lhs matrix
				for(int col1 : col1Arr) { // Varying col sizes for lhs matrix/row sizes for rhs matrix
					for(int col2 : col2Arr) { // Varying col sizes for rhs matrix
						// Generate two random dense matrices
						MatrixBlock m1 = MatrixBlock.randOperations(row, col1, sparsity1, -10, 10, "uniform", 7);
						MatrixBlock m2 = MatrixBlock.randOperations(col1, col2, sparsity2, -10, 10, "uniform", 8);
						MatrixBlock retScalar = new MatrixBlock(row, col2, false);
						MatrixBlock retSIMD = new MatrixBlock(row, col2, false);
						MatrixBlock retMKL = new MatrixBlock(row, col2, false);

						avg1 = 0;
						avg2 = 0;
						avg3 = 0;

						// Measure execution time for the scalar multiplication.
						for(int i = 0; i < 10; i++) {
							t1 = System.nanoTime();
							LibMatrixMult2.matrixMult(m1, m2, retScalar, k);
							avg1 += (System.nanoTime() - t1) / 1000000;
						}

						// Measure execution time for the SIMD multiplication.
						for(int i = 0; i < 10; i++) {
							t2 = System.nanoTime();
							LibMatrixMult.matrixMult(m1, m2, retSIMD, k); // SIMD
							avg2 += (System.nanoTime() - t2) / 1000000;
						}

						boolean isValidForNative = !LibMatrixNative.isMatMultMemoryBound(m1.getNumRows(),
							m1.getNumColumns(), m2.getNumColumns()) && !m1.isInSparseFormat() &&
							!m2.isInSparseFormat() &&
							(m1.getDenseBlock().isContiguous() || !LibMatrixNative.isSinglePrecision()) &&
							m2.getDenseBlock().isContiguous() // contiguous but not allocated
							&& 8L * retMKL.getLength() < Integer.MAX_VALUE;

						if(isValidForNative) {
							// Measure execution time for the MKL multiplication.
							for(int i = 0; i < 10; i++) {
								t3 = System.nanoTime();
								LibMatrixNative.matrixMult(m1, m2, retMKL, k);
								avg3 += (System.nanoTime() - t3) / 1000000;
							}
							avg3 /= 10;
							resEqual2 = compareResults(retSIMD, retMKL, row, col2);
						}

						avg1 /= 10;
						avg2 /= 10;
						resEqual1 = compareResults(retSIMD, retScalar, row, col2);
						improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

						printStats(sparsity1, sparsity2, row, col1, col1, col2, k, isValidForNative, false);

						// Write to csv
						if(isValidForNative) {
							writer.append(row + "," + col1 + "," + col2 + "," + k + "," + avg1 + "," + avg2 + "," + avg3
								+ "," + improvement + "," + (resEqual1 && resEqual2) + "\n");
						}
						else {
							writer.append(row + "," + col1 + "," + col2 + "," + k + "," + avg1 + "," + avg2 + "," + -1.0
								+ "," + improvement + "," + resEqual1 + "\n");
						}
					}
				}
			}
		}
		catch(IOException e) {
			e.printStackTrace();
		}
		catch(Exception e) { // better for grep
			System.out.println("ERROR in Mult: ");
			e.printStackTrace();
		}
	}

	public static void matrixDivTest(double sparsity1, double sparsity2, boolean sparseRet, String rows1, String cols1,
		String rows2, String cols2, int k, int warmupRuns) {

		try {
			Files.createDirectories(Paths.get(BASE_PATH));
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}

		int[] row1Arr = calculateSizes(rows1);
		int[] col1Arr = calculateSizes(cols1);
		int[] row2Arr = calculateSizes(rows2);
		int[] col2Arr = calculateSizes(cols2);

		String outputPath = getOutputPathDivTest(sparsity1, sparsity2, sparseRet, row2Arr[0], col2Arr[0], k);
		startWarmupDivTest(sparsity1, sparsity2, sparseRet, row1Arr[0], col1Arr[0], row2Arr[0], col2Arr[0], k, warmupRuns);

		try(FileWriter writer = new FileWriter(outputPath)) {
			// Write CSV header
			writer.append("rows1,cols1,rows2,cols2,k,time_scalar,time_simd,improvement,correctness\n");

			for(int row1 : row1Arr) { // Varying sizes for row lhs matrix
				for(int col1 : col1Arr) { // Varying sizes for col lhs matrix
					for(int row2 : row2Arr) { // Varying sizes for row rhs matrix
						for(int col2 : col2Arr) { // Varying sizes for col rhs matrix
							MatrixBlock m1 = MatrixBlock.randOperations(row1, col1, sparsity1, -10, 10, "uniform", 7);;
							MatrixBlock m2 = null;
							if(row2 == 1 || col2 == 1) {
								m2 = MatrixBlock.randOperations(row2, col2, sparsity2, -10, 10, "uniform", 24);
							} else {
								m2 = MatrixBlock.randOperations(row2, col2, sparsity2, -10, 10, "uniform", 24);
							}

							MatrixBlock retScalar, retSIMD;
							retScalar = new MatrixBlock(row1, col1, sparseRet);
							retSIMD = new MatrixBlock(row1, col1, sparseRet);

							avg1 = 0;
							avg2 = 0;

							for(int i = 0; i < 10; i++) {
								retScalar.reset(row1, col1, sparseRet);
								t1 = System.nanoTime();
								LibMatrixBincell.bincellOp(m1, m2, retScalar,
									new BinaryOperator(Divide.getDivideFnObject()), k);
								avg1 += (System.nanoTime() - t1) / 1000000;
							}

							for(int i = 0; i < 10; i++) {
								retSIMD.reset(row1, col1, sparseRet);
								t2 = System.nanoTime();
								LibMatrixBincell2.bincellOp(m1, m2, retSIMD,
									new BinaryOperator(Divide.getDivideFnObject()), k);
								avg2 += (System.nanoTime() - t2) / 1000000;
							}

							avg1 /= 10;
							avg2 /= 10;
							resEqual1 = compareResults(retScalar, retSIMD, row1, col1);
							improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

							printStats(sparsity1, sparsity2, row1, col1, row2, col2, k, false, false);

							// Write to csv
							writer.append(row1 + "," + col1 + "," + row2 + "," + col2 + "," + k + "," + avg1 + ","
								+ avg2 + "," + improvement + "," + resEqual1 + "\n");
						}
					}
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

	public static void matrixPowerTest(double sparsity, String rows, String cols, double exponent, int k,
		int warmupRuns) {

		try {
			Files.createDirectories(Paths.get(BASE_PATH));
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}

		int[] rowArr = calculateSizes(rows);
		int[] colArr = calculateSizes(cols);

		String outputPath = getOutputPathPowerTest(sparsity, exponent, k);
		startWarmupPowerTest(sparsity, rowArr[0], colArr[0], exponent, k, warmupRuns);

		try(FileWriter writer = new FileWriter(outputPath)) {
			// Write CSV header
			writer.append("rows,cols,k,time_scalar,time_simd,improvement,correctness\n");

			for(int row : rowArr) { // Varying sizes for row
				for(int col : colArr) { // Varying sizes for col
					MatrixBlock m1 = MatrixBlock.randOperations(row, col, sparsity, -10, 10, "uniform", 7);
					MatrixBlock retScalar = new MatrixBlock(row, col, false);
					MatrixBlock retSIMD = new MatrixBlock(row, col, false);
					RightScalarOperator powerOpK = new RightScalarOperator(Power.getPowerFnObject(), exponent, k);

					avg1 = 0;
					avg2 = 0;

					for(int i = 0; i < 10; i++) {
						retScalar.reset(row, col, 0);
						t1 = System.nanoTime();
						retScalar = m1.scalarOperations(powerOpK, new MatrixBlock());
						avg1 += (System.nanoTime() - t1) / 1000000;
					}

					for(int i = 0; i < 10; i++) {
						retSIMD.reset(row, col, 0);
						t2 = System.nanoTime();
						retSIMD = m1.scalarOperationsSIMD(powerOpK, new MatrixBlock());
						avg2 += (System.nanoTime() - t2) / 1000000;
					}

					avg1 /= 10;
					avg2 /= 10;
					resEqual1 = compareResults(retScalar, retSIMD, row, col);
					improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

					printStats(sparsity, 0.0, row, col, 0, 0, k, false, true);

					// Write to csv
					writer.append(row + "," + col + "," + k + "," + avg1 + "," + avg2 + "," + improvement + "," + resEqual1 + "\n");
				}
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

	public static void matrixExpTest(double sparsity, String rows, String cols, int k, int warmupRuns) {

		try {
			Files.createDirectories(Paths.get(BASE_PATH));
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}

		int[] rowArr = calculateSizes(rows);
		int[] colArr = calculateSizes(cols);

		String outputPath = getOutputPathExpTest(sparsity, k);
		startWarmupExpTest(sparsity, rowArr[0], colArr[0], k, warmupRuns);

		try(FileWriter writer = new FileWriter(outputPath)) {
			// Write CSV header
			writer.append("rows,cols,k,time_scalar,time_simd,improvement,correctness\n");

			for(int row : rowArr) { // Varying sizes for row
				for(int col : colArr) { // Varying sizes for col
					MatrixBlock m1 = MatrixBlock.randOperations(row, col, sparsity, -10, 10, "uniform", 7);
					MatrixBlock retScalar = new MatrixBlock(row, col, false);
					MatrixBlock retSIMD = new MatrixBlock(row, col, false);

					UnaryOperator expOperator = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.EXP), k);

					avg1 = 0;
					avg2 = 0;

					for(int i = 0; i < 10; i++) {
						retScalar.reset(row, col, 0);
						t1 = System.nanoTime();
						retScalar = m1.unaryOperations(expOperator, new MatrixBlock());
						avg1 += (System.nanoTime() - t1) / 1000000;
					}

					for(int i = 0; i < 10; i++) {
						retSIMD.reset(row, col, 0);
						t2 = System.nanoTime();
						retSIMD = m1.unaryOperationsSIMD(expOperator, new MatrixBlock());
						avg2 += (System.nanoTime() - t2) / 1000000;
					}

					avg1 /= 10;
					avg2 /= 10;
					resEqual1 = compareResults(retScalar, retSIMD, row, col);
					improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

					printStats(sparsity, 0.0, row, col, 0, 0, k, false, true);

					// Write to csv
					writer.append(row + "," + col + "," + k + "," + avg1 + "," + avg2 + "," + improvement + "," + resEqual1 + "\n");
				}
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
		MatrixBlock ret = new MatrixBlock(rows, cols2, false);

		boolean isValidForNative = !LibMatrixNative.isMatMultMemoryBound(m1.getNumRows(), m1.getNumColumns(),
			m2.getNumColumns()) && !m1.isInSparseFormat() && !m2.isInSparseFormat() &&
			(m1.getDenseBlock().isContiguous() || !LibMatrixNative.isSinglePrecision()) &&
			m2.getDenseBlock().isContiguous() // contiguous but not allocated
			&& 8L * ret.getLength() < Integer.MAX_VALUE;

		for(int i = 0; i < warmupRuns; i++) {
			LibMatrixMult2.matrixMult(m1, m2, k);
			LibMatrixMult.matrixMult(m1, m2, k);
			if(isValidForNative)
				LibMatrixNative.matrixMult(m1, m2, ret, k);
		}
	}

	private static void startWarmupDivTest(double sparsity1, double sparsity2, boolean sparseRet, int rows1, int cols1,
		int rows2, int cols2, int k, int warmupRuns) {
		MatrixBlock m1 = MatrixBlock.randOperations(rows1, cols1, sparsity1, -10, 10, "uniform", 7);
		MatrixBlock m2 = MatrixBlock.randOperations(rows2, cols2, sparsity2, -10, 10, "uniform", 8);
		MatrixBlock ret = new MatrixBlock(rows1, cols1 , sparseRet);

		for(int i = 0; i < warmupRuns; i++) {
			LibMatrixBincell.bincellOp(m1, m2, ret, new BinaryOperator(Divide.getDivideFnObject(), k));
			LibMatrixBincell2.bincellOp(m1, m2, ret, new BinaryOperator(Divide.getDivideFnObject(), k));
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

		for(int i = 0; i < warmupRuns; i++) {
			m1.unaryOperations(expOperator, new MatrixBlock());
			m1.unaryOperationsSIMD(expOperator, new MatrixBlock());
		}
	}

	private static String getOutputPathMultTest(double sparsity1, double sparsity2, int rows1, int cols1, int rows2, int cols2, int k) {
		String prefix = "INVALID_";
		if(sparsity1 >= 0.4 && sparsity2 >= 0.4) { // DENSE DENSE
			if(rows1 > 17 && cols1 > 1 && cols2 > 1) {
				prefix = "perfMultDenseDenseMM_";
			}
			if(rows2 <= 2*1024 && cols2 == 1) {
				prefix = "perfMultDenseDenseMVShortRHS_";
			}
		} else if(sparsity1 < 0.4 && sparsity2 >= 0.4) { // SPARSE DENSE
			if(rows1 > 17 && cols1 > 1 && cols2 > 1) {
				prefix = "perfMultSparseDenseMM_";
			}
			if(rows2 <= 2*1024 && cols2 == 1) {
				prefix = "perfMultSparseDenseMVShortRHS_";
			}
		} else if(sparsity1 >= 0.4 && sparsity2 < 0.4) { // DENSE SPARSE
			prefix = "perfMultDenseSparse_";
		}
		return BASE_PATH + prefix + getTimeStamp() + "_s1=" + sparsity1 + "_s2=" + sparsity2 + "_k=" + k + ".csv";
	}

	private static String getOutputPathDivTest(double sparsity1, double sparsity2, boolean sparseRet, int rows2, int cols2, int k) {
		String prefix = "INVALID_";
		if(sparsity2 < 1.0)
			return prefix;

		if(sparsity1 >= 0.4 && !sparseRet) {
			if(rows2 == 1 && cols2 > 1) {
				prefix = "perfSafeBinaryMVDenseRowVector_";
			} else if(rows2 > 1 && cols2 == 1) {
				prefix = "perfSafeBinaryMVDenseColVector_";
			} else if(rows2 > 1 && cols2 > 1) {
				prefix = "perfSafeBinaryMMDenseDenseDense_";
			}
		} else if(sparsity1 < 0.4) {
			if(sparseRet && rows2 > 1 && cols2 == 1) { // Check necessary because of potential invocation of safeBinaryMVSparseDenseRow
				prefix = "perfSafeBinaryMVSparseColVector_";
			} else if(sparseRet && rows2 == 1 && cols2 > 1) {
				prefix = "perfSafeBinaryMVSparseRowVector_";
			} else if(sparseRet && rows2 > 1 && cols2 > 1) {
				prefix = "perfSafeBinaryMMSparseDenseSkip_";
			}
		}
		return BASE_PATH + prefix + getTimeStamp() + "_s1=" + sparsity1 + "_s2=" + sparsity2 + "_s3=" + sparseRet
			+ "_k=" + k + ".csv";
	}

	private static String getOutputPathPowerTest(double sparsity1, double exponent, int k) {
		String prefix = sparsity1 < 0.4 ? "perfPowerSparse_" : "perfPowerDense_";
		return BASE_PATH + "perfPower_" + getTimeStamp() + "_s1=" + sparsity1 + "_exponent=" + exponent + "_k=" + k
			+ ".csv";
	}

	private static String getOutputPathExpTest(double sparsity1, int k) {
		String prefix = sparsity1 < 0.4 ? "perfExpSparse_" : "perfExpDense_";
		return BASE_PATH + prefix + getTimeStamp() + "_s1=" + sparsity1 + "_k=" + k + ".csv";
	}

	private static String getTimeStamp() {
		LocalDateTime now = LocalDateTime.now();
		DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss");
		return now.format(formatter);
	}

	private static void printStats(double sparsity1, double sparsity2, int rows1, int cols1, int rows2, int cols2, int k, boolean mklUsed,
		boolean singleMatrixOp) {
		String statImprovement = "Improvement scalar -> SIMD: " + improvement + "%";
		String statAverages = mklUsed ? "Averages - Scalar: " + avg1 + "; SIMD: " + avg2 + "; MKL: "
			+ avg3 : "Averages - Scalar: " + avg1 + "; SIMD: " + avg2;
		String statThreadsAndSparsities = "Threads: " + k
			+ (singleMatrixOp ? "; Sparsity: " + sparsity1 : "; Sparsity1: " + sparsity1 + "; Sparsity2: " + sparsity2);
		String statMatrices = singleMatrixOp ? rows1 + "x" + cols1 : "LHS: " + rows1 + "x" + cols1 + "; RHS: " + rows2
			+ "x" + cols2;
		String statResultsEqual = "Results are equal: " + ((mklUsed) ? (resEqual1 && resEqual2) : resEqual1);

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

	private static int[] calculateSizes(String stepStr) {
		String[] strArr = stepStr.split("-");

		if(strArr.length == 1) {
			// stepStr only contains 1 value
			return new int[] {Integer.parseInt(strArr[0])};
		}

		// Continue with split string
		String[] strArr2 = strArr[1].split("#");
		int start = Integer.valueOf(strArr[0]);
		int end = Integer.valueOf(strArr2[0]);
		int step = Integer.valueOf(strArr2[1]);

		int n = ((int) (end - start) / step) + 1;

		int[] sizes = new int[n];
		for(int i = 0; i < sizes.length; i++) {
			sizes[i] = start + step * i;
		}
		return sizes;
	}
}