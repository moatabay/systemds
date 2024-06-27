package org.apache.sysds.performance.matrix;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Divide;
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

public class SIMDPerformance {

	private static final String BASE_PATH = "vector_api_test/";
	private static final double EPSILON = 1E-9;

	private static long t1, t2, t3;
	private static double avg1, avg2, avg3, improvement;
	private static boolean resEqual1, resEqual2;

	public static void matrixMultTest(double sparsity1, double sparsity2, String rows1, String cols1, String cols2,
		int k, int warmupRuns, String dmlPath) {
		createBasePath();

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
							LibMatrixMult.matrixMult(m1, m2, retScalar, k);
							avg1 += (System.nanoTime() - t1) / 1_000_000_000.0;
						}

						// Measure execution time for the SIMD multiplication.
						for(int i = 0; i < 10; i++) {
							t2 = System.nanoTime();
							LibMatrixMultSIMD.matrixMult(m1, m2, retSIMD, k); // SIMD
							avg2 += (System.nanoTime() - t2) / 1_000_000_000.0;
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
								avg3 += (System.nanoTime() - t3) / 1_000_000_000.0;
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

	public static void matrixDivTest(double sparsity1, double sparsity2, double sparsity3, String rows1, String cols1,
		int mode, int k, int warmupRuns) {
		createBasePath();

		int[] row1Arr = calculateSizes(rows1);
		int[] col1Arr = calculateSizes(cols1);
		boolean sparseRet = sparsity3 < 0.4;

		String outputPath = getOutputPathDivTest(sparsity1, sparsity2, sparsity3, mode, k);
		startWarmupDivTest(sparsity1, sparsity2, sparseRet, row1Arr[0], col1Arr[0], mode, k, warmupRuns);

		try(FileWriter writer = new FileWriter(outputPath)) {
			// Write CSV header
			writer.append("rows1,cols1,k,time_scalar,time_simd,improvement,correctness\n");

			for(int row1 : row1Arr) { // Varying sizes for row lhs matrix
				for(int col1 : col1Arr) { // Varying sizes for col lhs matrix
					MatrixBlock m1 = MatrixBlock.randOperations(row1, col1, sparsity1, -10, 10, "uniform", 7);;
					MatrixBlock m2 = null;
					if(mode == 0) // matrix-matrix
						m2 = MatrixBlock.randOperations(row1, col1, sparsity2, -10, 10, "uniform", 24);
					else if(mode == 1) // matrix-row vector
						m2 = MatrixBlock.randOperations(1, col1, sparsity2, -10, 10, "uniform", 24);
					else if(mode == 2) // matrix-col vector
						m2 = MatrixBlock.randOperations(row1, 1, sparsity2, -10, 10, "uniform", 24);

					MatrixBlock retScalar, retSIMD;
					retScalar = new MatrixBlock(row1, col1, sparseRet);
					retSIMD = new MatrixBlock(row1, col1, sparseRet);

					avg1 = 0;
					avg2 = 0;

					for(int i = 0; i < 20; i++) {
						retScalar.reset(row1, col1, sparseRet);
						t1 = System.nanoTime();
						LibMatrixBincell.bincellOp(m1, m2, retScalar, new BinaryOperator(Divide.getDivideFnObject()), k);
						avg1 += (System.nanoTime() - t1) / 1_000_000_000.0;
					}

					for(int i = 0; i < 20; i++) {
						retSIMD.reset(row1, col1, sparseRet);
						t2 = System.nanoTime();
						LibMatrixBincellSIMD.bincellOp(m1, m2, retSIMD, new BinaryOperator(Divide.getDivideFnObject()), k);
						avg2 += (System.nanoTime() - t2) / 1_000_000_000.0;
					}

					avg1 /= 20;
					avg2 /= 20;
					resEqual1 = compareResults(retScalar, retSIMD, row1, col1);
					improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

					printStats(sparsity1, sparsity2, row1, col1, row1, col1, k, false, false);

					// Write to csv
					writer.append(row1 + "," + col1 + "," + k + "," + avg1 + ","
							+ avg2 + "," + improvement + "," + resEqual1 + "\n");
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
		int warmupRuns, boolean useSpecies256) {
		createBasePath();

		int[] rowArr = calculateSizes(rows);
		int[] colArr = calculateSizes(cols);

		String outputPath = getOutputPathPowerTest(sparsity, exponent, k);
		startWarmupPowerTest(sparsity, rowArr[0], colArr[0], exponent, k, warmupRuns);

		try(FileWriter writer = new FileWriter(outputPath)) {
			// Write CSV header
			if (useSpecies256) {
				writer.append("rows,cols,k,time_scalar,time_simd,time_simd_256,improvement,correctness\n");
			} else {
				writer.append("rows,cols,k,time_scalar,time_simd,improvement,correctness\n");
			}

			for(int row : rowArr) { // Varying sizes for row
				for(int col : colArr) { // Varying sizes for col
					MatrixBlock m1 = MatrixBlock.randOperations(row, col, sparsity, -10, 10, "uniform", 7);
					MatrixBlock retScalar = new MatrixBlock(row, col, false);
					MatrixBlock retSIMD = new MatrixBlock(row, col, false);
					MatrixBlock retSIMD256 = new MatrixBlock(row, col, false);
					RightScalarOperator powerOpK = new RightScalarOperator(Power.getPowerFnObject(), exponent, k);

					avg1 = 0;
					avg2 = 0;
					avg3 = 0;

					for(int i = 0; i < 10; i++) {
						retScalar.reset(row, col, 0);
						t1 = System.nanoTime();
						retScalar = m1.scalarOperations(powerOpK, new MatrixBlock());
						avg1 += (System.nanoTime() - t1) / 1_000_000_000.0;
					}

					for(int i = 0; i < 10; i++) {
						retSIMD.reset(row, col, 0);
						t2 = System.nanoTime();
						retSIMD = m1.scalarOperationsSIMD(powerOpK, new MatrixBlock());
						avg2 += (System.nanoTime() - t2) / 1_000_000_000.0;
					}

					if(useSpecies256) {
						for(int i = 0; i < 10; i++) {
							retSIMD256.reset(row, col, 0);
							t3 = System.nanoTime();
							retSIMD256 = m1.scalarOperationsSIMD256(powerOpK, new MatrixBlock());
							avg3 += (System.nanoTime() - t3) / 1_000_000_000.0;
						}
						avg3 /= 10;
					}

					avg1 /= 10;
					avg2 /= 10;

					resEqual1 = compareResults(retScalar, retSIMD, row, col);
					improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

					printStats(sparsity, 0.0, row, col, 0, 0, k, false, true);

					// Write to csv
					if (useSpecies256) {
						writer.append(row + "," + col + "," + k + "," + avg1 + "," + avg2 + "," + avg3 + "," + improvement + "," + resEqual1 + "\n");
					} else {
						writer.append(row + "," + col + "," + k + "," + avg1 + "," + avg2 + "," + improvement + "," + resEqual1 + "\n");
					}
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
		createBasePath();

		int[] rowArr = calculateSizes(rows);
		int[] colArr = calculateSizes(cols);

		String outputPath = getOutputPathExpTest(sparsity, k);
		startWarmupExpTest(sparsity, rowArr[0], colArr[0], k, warmupRuns);

		try(FileWriter writer = new FileWriter(outputPath)) {
			// Write CSV header
			writer.append("rows,cols,k,time_scalar,time_simd,time_scalar2,improvement,correctness\n");

			for(int row : rowArr) { // Varying sizes for row
				for(int col : colArr) { // Varying sizes for col
					MatrixBlock m1 = MatrixBlock.randOperations(row, col, sparsity, -10, 10, "uniform", 7);
					MatrixBlock retScalar = new MatrixBlock(row, col, false);
					MatrixBlock retScalar2 = new MatrixBlock(row, col, false);
					MatrixBlock retSIMD = new MatrixBlock(row, col, false);

					UnaryOperator expOperator = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.EXP), k);
					UnaryOperator exp2Operator = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.EXP_2), k);

					avg1 = 0;
					avg2 = 0;
					avg3 = 0;

					for(int i = 0; i < 20; i++) {
						retScalar.reset(row, col, 0);
						t1 = System.nanoTime();
						retScalar = m1.unaryOperations(expOperator, new MatrixBlock());
						avg1 += (System.nanoTime() - t1) / 1_000_000_000.0;
					}

					for(int i = 0; i < 20; i++) {
						retSIMD.reset(row, col, 0);
						t2 = System.nanoTime();
						retSIMD = m1.unaryOperationsSIMD(expOperator, new MatrixBlock());
						avg2 += (System.nanoTime() - t2) / 1_000_000_000.0;
					}

					for(int i = 0; i < 20; i++) {
						retScalar2.reset(row, col, 0);
						t3 = System.nanoTime();
						retScalar2 = m1.unaryOperations(exp2Operator, new MatrixBlock());
						avg3 += (System.nanoTime() - t3) / 1_000_000_000.0;
					}

					avg1 /= 20;
					avg2 /= 20;
					avg3 /= 20;
					resEqual1 = compareResults(retScalar, retSIMD, row, col);
					improvement = Math.round(((avg1 - avg2) / avg1) * 100.0 * 100.0) / 100.0;

					printStats(sparsity, 0.0, row, col, 0, 0, k, false, true);

					// Write to csv
					writer.append(row + "," + col + "," + k + "," + avg1 + "," + avg2 + "," + avg3 + "," + improvement + "," + resEqual1 + "\n");
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


/*
		mkl is not utilized when:
		- vectors are involved AND if one of the matrices/vectors exceeds a certain size.
		- if m1 or m2 is sparse
		- if m1 is non-contiguous
		- if m2 is non-contiguous
		- if 8 * ret.getLength >= Integer.MAX_VALUE
*/
		boolean isValidForNative = !LibMatrixNative.isMatMultMemoryBound(m1.getNumRows(), m1.getNumColumns(),
			m2.getNumColumns()) && !m1.isInSparseFormat() && !m2.isInSparseFormat() &&
			(m1.getDenseBlock().isContiguous() || !LibMatrixNative.isSinglePrecision()) &&
			m2.getDenseBlock().isContiguous() // contiguous but not allocated
			&& 8L * ret.getLength() < Integer.MAX_VALUE;

		for(int i = 0; i < warmupRuns; i++) {
			LibMatrixMult.matrixMult(m1, m2, k);
			LibMatrixMultSIMD.matrixMult(m1, m2, k);
			if(isValidForNative)
				LibMatrixNative.matrixMult(m1, m2, ret, k);
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
		MatrixBlock ret = new MatrixBlock(rows1, cols1 , sparseRet);

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
			m1.scalarOperationsSIMD256(powerOpK, new MatrixBlock());
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

	private static String getOutputPathMultTest(double sparsity1, double sparsity2, int rows1, int cols1, int rows2, int cols2, int k) {
		String prefix = "INVALID_";
		if(sparsity1 >= 0.4 && sparsity2 >= 0.4) { // DENSE DENSE
			if(rows1 > 17 && cols1 > 1 && cols2 > 1) {
				prefix = "ddmmmult_";
			}
			if(rows2 <= 2*1024 && cols2 == 1) {
				prefix = "ddmvmult_";
			}
		} else if(sparsity1 < 0.4 && sparsity2 >= 0.4) { // SPARSE DENSE
			if(rows1 > 17 && cols1 > 1 && cols2 > 1) {
				prefix = "sdmmmult_";
			}
			if(rows2 <= 2*1024 && cols2 == 1) {
				prefix = "sdmvmult_";
			}
		} else if(sparsity1 >= 0.4 && sparsity2 < 0.4) { // DENSE SPARSE
			prefix = "dsmmmult_";
		}
		return BASE_PATH + prefix + getTimeStamp() + "_s1=" + sparsity1 + "_s2=" + sparsity2 + "_k=" + k + ".csv";
	}

	private static String getOutputPathDivTest(double sparsity1, double sparsity2, double sparsity3, int mode, int k) {
		String prefix = "INVALID_";

		if(sparsity1 >= 0.4 && sparsity2 >= 0.4 && sparsity3 >= 0.4) { // ALL DENSE
			if(mode == 0)
				prefix = "ddmmdiv_";
			else if(mode == 1)
				prefix = "ddmvdiv_row_";
			else if(mode == 2)
				prefix = "ddmvdiv_col_";
		} else if(sparsity1 < 0.4 && sparsity2 >= 0.4) {
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

	private static String getOutputPathPowerTest(double sparsity1, double exponent, int k) {
		String prefix = sparsity1 < 0.4 ? "sparse_pow_" : "dense_pow_";
		return BASE_PATH + prefix + getTimeStamp() + "_s1=" + sparsity1 + "_exponent=" + exponent + "_k=" + k
			+ ".csv";
	}

	private static String getOutputPathExpTest(double sparsity1, int k) {
		String prefix = sparsity1 < 0.4 ? "sparse_exp_" : "dense_exp_";
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
		int mode = 0;

		if (strArr.length == 1) {
			// stepStr only contains 1 value
			return new int[]{Integer.parseInt(strArr[0])};
		}

		// Continue with split string
		String[] strArr2 = null;
		if (strArr[1].contains("#")) {
			strArr2 = strArr[1].split("#");
		} else if (strArr[1].contains(":")) {
			strArr2 = strArr[1].split(":");
			mode = 1;
		}

		int start = Integer.parseInt(strArr[0]);
		int end = Integer.parseInt(strArr2[0]);
		int step = Integer.parseInt(strArr2[1]);

		int n = (mode == 0)
				? ((end - start) / step) + 1
				: (int) (Math.log10(end / start) / Math.log10(step)) + 1;

		int[] sizes = new int[n];
		int i = 0;

		if (mode == 0) {
			for (; i < sizes.length; i++) {
				sizes[i] = start + step * i;
			}
		} else {
			for (; i < sizes.length; i++) {
				sizes[i] = (int) (start * Math.pow(step, i));
			}
		}
		return sizes;
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
