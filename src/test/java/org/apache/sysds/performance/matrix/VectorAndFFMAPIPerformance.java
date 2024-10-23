package org.apache.sysds.performance.matrix;

import static org.apache.sysds.performance.matrix.MatrixSIMDPerformance.matrixMultTest;

public class VectorAndFFMAPIPerformance {

	// Dummy data used for testing
	private static double denseSparsity = 0.8; // Only one dense value necessary
	private static double denseFull = 1.0; // Important for sparse-safe div
	private static double[] sparseSparsities = {0.398, 0.2, 0.1, 0.01, 0.001};
	private static int[] threads = {1, Runtime.getRuntime().availableProcessors()}; // 1 or maximum amount of threads
	private static double[] exponents = {1.5, 20, 160, 200.18322, 308}; // Biggest double: 1.7976931348623157E308

	/**
	 * Starts the mega-tests for matrix-multiplication.
	 * @param sizesRow Rows of LHS matrix
	 * @param sizesCol1 Columns of LHS matrix/Rows of RHS matrix
	 * @param sizesCol2 Columns of RHS matrix
	 * @param mode Int value that specifies the matrix-mult method to run. Mode 0 = All Matrix-Matrix, 1 All Matrix-Vector.
	 * @param warumupRuns Amount of warm up runs for JIT optimization
	 * @param dmlPath Path to the SystemDS-config.xml for native BLAS/MKL calls
	 */
	public static void runMatrixMultTests(String sizesRow, String sizesCol1, String sizesCol2, int mode, int warumupRuns, String dmlPath) {
		// Modes = 0 == MM
		// Test each method
		if(mode == 0 || mode == 2) { // matrixMultDenseDenseMM
			for(int k : threads) {
				matrixMultTest(denseSparsity, denseSparsity, sizesRow, sizesCol1, sizesCol2, k, warumupRuns, dmlPath);
			}
		}
		if(mode == 0 || mode == 3) { // matrixMultDenseSparse
			for(int k : threads) {
				for(double sparsity : sparseSparsities) {
					matrixMultTest(denseSparsity, sparsity, sizesRow, sizesCol1, sizesCol2, k, warumupRuns, dmlPath);
				}
			}
		}
		if(mode == 0 || mode == 4) { // matrixMultSparseDenseMM
			for(int k : threads) {
				for(double sparsity : sparseSparsities) {
					matrixMultTest(sparsity, denseSparsity, sizesRow, sizesCol1, sizesCol2, k, warumupRuns, dmlPath);
				}
			}
		}
		if(mode == 1 || mode == 5) { // matrixMultDenseDenseMVShortRHS
			for(int k : threads) {
				matrixMultTest(denseSparsity, denseSparsity, sizesRow, sizesCol1, "1", k, warumupRuns, dmlPath);
			}
		}
		if(mode == 1 || mode == 6) { // matrixMultSparseDenseMVShortRHS
			for(int k : threads) {
				for(double sparsity : sparseSparsities) {
					matrixMultTest(sparsity, denseSparsity, sizesRow, sizesCol1, "1", k, warumupRuns, dmlPath);
				}
			}
		}
	}

	/**
	 * Starts the mega-tests for element-wise division.
	 * @param sizesRow Rows of matrices
	 * @param sizesCol Columns of matrices
	 * @param mode Int value that specifies the div method to run. Full documentation in method body
	 * @param warumupRuns Amount of warm up runs for JIT optimization
	 */
	public static void runMatrixDivTests(String sizesRow, String sizesCol, int mode, int warumupRuns) {
		/*
		Mode 0 = safeBinaryMMDenseDenseDense and safeBinaryMMSparseDenseSkip
		Mode 1 = safeBinaryMVDense Col/Row Vector
		Mode 2 = safeBinaryMVSparse Col/Row Vector
		 */
		// Test each method
		if(mode == 0 || mode == 3) { // safeBinaryMMDenseDenseDense
			for(int k : threads) {
				MatrixSIMDPerformance.matrixDivTest(denseSparsity, denseFull, false, sizesRow, sizesCol, sizesRow, sizesCol, k, warumupRuns);
			}
		}
		if(mode == 0 || mode == 4) { // safeBinaryMMSparseDenseSkip
			for(int k : threads) {
				for(double sparsity : sparseSparsities) {
					MatrixSIMDPerformance.matrixDivTest(sparsity, denseFull, false, sizesRow, sizesCol, sizesRow, sizesCol, k, warumupRuns);
				}
			}
		}
		if(mode == 1 || mode == 5) { // safeBinaryMVDense -> Col Vector
			for(int k : threads) {
				MatrixSIMDPerformance.matrixDivTest(denseSparsity, denseFull, false, sizesRow, sizesCol, sizesRow, "1", k, warumupRuns); // TODO: when sparseRet = true -> better performance?
			}
		}
		if(mode == 1 || mode == 6) { // safeBinaryMVDense -> Row Vector
			for(int k : threads) {
				MatrixSIMDPerformance.matrixDivTest(denseSparsity, denseFull, false, sizesRow, sizesCol, "1", sizesCol, k, warumupRuns);
			}
		}
		if(mode == 2 || mode == 7) { // safeBinaryMVSparse -> Col Vector
			for(int k : threads) {
				for(double sparsity : sparseSparsities) {
					MatrixSIMDPerformance.matrixDivTest(sparsity, denseFull, false, sizesRow, sizesCol, sizesRow, "1", k, warumupRuns); // TODO: when sparseRet = true -> better performance?
				}
			}
		}
		if(mode == 2 || mode == 8) { // safeBinaryMVSparse -> Row Vector
			for(int k : threads) {
				for(double sparsity : sparseSparsities) {
					MatrixSIMDPerformance.matrixDivTest(sparsity, denseFull, true, sizesRow, sizesCol, "1", sizesCol, k, warumupRuns);
				}
			}
		}
	}

	/**
	 * Starts the mega-tests for element-wise power.
	 * @param sizesRow Rows of matrices
	 * @param sizesCol Columns of matrices
	 * @param warumupRuns Amount of warm up runs for JIT optimization
	 */
	public static void runMatrixPowerTests(String sizesRow, String sizesCol, int mode, int warumupRuns) {
		if(mode == 0 || mode == 1) { // Dense Power
			for(int k : threads) {
				for(double exponent : exponents) {
					MatrixSIMDPerformance.matrixPowerTest(denseSparsity, sizesRow, sizesCol, exponent, k, warumupRuns);
				}
			}
		}

		if(mode == 0 || mode == 2) { // Sparse Power
			for(int k : threads) {
				for(double sparsity : sparseSparsities) {
					for(double exponent : exponents) {
						MatrixSIMDPerformance.matrixPowerTest(denseSparsity, sizesRow, sizesCol, exponent, k, warumupRuns);
					}
				}
			}
		}
	}

	/**
	 * Starts the mega-tests for element-wise division.
	 * @param sizesRow Rows of matrices
	 * @param mode
	 * @param sizesCol Columns of matrices
	 * @param warumupRuns Amount of warm up runs for JIT optimization
	 */
	public static void runMatrixExpTests(String sizesRow, String sizesCol, int mode, int warumupRuns) {
		if(mode == 0 || mode == 1) { // Dense Exp (Single threaded and multithreaded implementations are different)
			for(int k : threads) {
				MatrixSIMDPerformance.matrixExpTest(denseSparsity, sizesRow, sizesCol, k, warumupRuns);
			}
		}

		// Sparse
		if(mode == 0 || mode == 2) { // Sparse Exp
			for(int k : threads) {
				for(double sparsity : sparseSparsities) {
					MatrixSIMDPerformance.matrixExpTest(sparsity, sizesRow, sizesCol, k, warumupRuns);
				}
			}
		}
	}

	public static void runFFMTests() {
		// TODO
	}
}
