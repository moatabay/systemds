package org.apache.sysds.performance.matrix;

import static org.apache.sysds.performance.matrix.MatrixSIMDPerformance.matrixMultTest;

public class VectorAndFFMAPIPerformance {

	private static double denseSparsity = 0.7;
	private static double[] sparseSparsities = {0.398, 0.2, 0.1, 0.01, 0.001};
	private static int[] threads = {1, Runtime.getRuntime().availableProcessors()}; // Get maximum amount of threads
	private static int[] exponents = {2, 20, 200, 308}; // 308 too much?

	public static void runMatrixMultTests(String dmlPath) {
		/*String sizesMM = "5000-12000#1000"; // TODO: remove soon
		String sizesMV = "200000-1600000#200000";
		int warmupRunsSm = 100;
		// Dense Dense MM
		for(int k : threads) {
			matrixMultTest(denseSparsity, denseSparsity, sizesSm, sizesSm, sizesSm, 32, 100, dmlPath);
		}

		// Dense Sparse MM
		for(int k : threads) {
			for(double sparseSparsity : sparseSparsities) {
				matrixMultTest(denseSparsity, 1.0, sizesSm, sizesSm, sizesSm, 32, 100, dmlPath);
			}
		}

		// Sparse Dense MM
		for(int k : threads) {
			for(double sparseSparsity : sparseSparsities) {
				// TODO: adjust
				matrixMultTest(denseSparsity, 1.0, sizesSm, sizesSm, sizesSm, 32, 100, dmlPath);
			}
		}

		// Dense Dense MV
		for(int k : threads) {
			// TODO: adjust
			matrixMultTest(denseSparsity, denseSparsity, sizesSm, sizesSm, sizesSm, 32, 100, dmlPath);
		}

		// Sparse Dense MV
		for(int k : threads) {
			for(double sparseSparsity : sparseSparsities) {
				// TODO: adjust
				matrixMultTest(denseSparsity, 1.0, sizesSm, sizesSm, sizesSm, 32, 100, dmlPath);
			}
		}*/
	}

	public static void runMatrixDivTests() {
		// TODO
	}

	public static void runMatrixPowerTests() {
		// TODO
		// exponent = 308 should be the hard cap, my matrix values are between -10 and 10
	}

	public static void runMatrixExpTests() {
		// TODO
	}

	public static void runFFMTests() {
		// TODO
	}
}
