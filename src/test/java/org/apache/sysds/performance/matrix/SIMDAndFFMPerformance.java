package org.apache.sysds.performance.matrix;

import static org.apache.sysds.performance.matrix.SIMDPerformance.*;

public class SIMDAndFFMPerformance {

	// Dummy data used for testing
	private static double denseSp = 0.8; // Only one dense value necessary
	private static double[] denseSpArray = {0.8};
	private static double[] sparsitiesLinear = {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35};
	private static double[] sparsitiesLinear2 = {0.001,0.0592,0.1174,0.1756,0.2338,0.292,0.35};
	private static double[] sparsitiesExponential = {0.0003, 0.003, 0.03, 0.3};

	private static int[] threads = {1, Runtime.getRuntime().availableProcessors()}; // 1 or maximum amount of threads
	private static double[] exponents = {2, 2.833};

	public static void runDenseVectorizedMultTests(String dmlPath) {
		System.out.println("runDenseVectorizedMultTests");

		for(int k : threads) { // dense dense MM
			matrixMultTest(denseSp,denseSp,"10000-70000#10000","2000","2000",k,10,dmlPath); // varying lhs row
			matrixMultTest(denseSp,denseSp,"2000","10000-70000#10000","2000",k,10,dmlPath); // varying lhs col
			matrixMultTest(denseSp,denseSp,"2000","2000","10000-70000#10000",k,10,dmlPath); // varying rhs col
		}

		for(int k : threads) { // dense dense MV
			matrixMultTest(denseSp,denseSp,"1000-10000000:10","500","1",k,100,dmlPath); // varying lhs row, 20 row vector
		}
	}

	public static void runDenseVectorizedElementWiseTests() {
		System.out.println("runDenseVectorizedElementWiseTests");
		for(int k : threads) { // dense exp
			matrixExpTest(denseSp, "10000-70000#10000", "5000", k, 15); // varying row
			matrixExpTest(denseSp, "5000", "10000-70000#10000", k, 15); // varying col
		}

		for(int k : threads) { // dense power
			for(double exponent : exponents) {
				matrixPowerTest(denseSp,"10000-70000#10000", "5000",exponent,k,40, false); // varying row
				matrixPowerTest(denseSp,"5000", "10000-70000#10000",exponent,k,40, false); // varying col
			}
		}

		for(int k : threads) { // dense div MM
			matrixDivTest(denseSp,denseSp,denseSp,"10000-70000#10000","5000",0,k,15); // varying row
			matrixDivTest(denseSp,denseSp,denseSp,"5000","10000-70000#10000",0,k,15); // varying col
		}

		for(int k : threads) { // dense div MV - row vector
			matrixDivTest(denseSp,denseSp,denseSp,"10000-70000#10000","5000",1,k,15); // varying row
			matrixDivTest(denseSp,denseSp,denseSp,"5000","10000-70000#10000",1,k,15); // varying col
		}

		for(int k : threads) { // dense div MV - col vector
			matrixDivTest(denseSp,denseSp,denseSp,"10000-70000#10000","5000",2,k,15); // varying row
			matrixDivTest(denseSp,denseSp,denseSp,"5000","10000-70000#10000",2,k,15); // varying col
		}
	}

	public static void runSparseVectorizedMultTests(String dmlPath) {
		System.out.println("runSparseVectorizedMultTests");

		for(int k : threads) { // dense sparse MM
			SIMDSparsePerformance.matrixMultTest(denseSpArray, sparsitiesLinear, 2000, 5000, 50000, k, 10);
			SIMDSparsePerformance.matrixMultTest(denseSpArray, sparsitiesExponential, 2000, 5000, 100000, k, 10);
		}

		for(int k : threads) { // sparse dense MM
			SIMDSparsePerformance.matrixMultTest(sparsitiesLinear2, denseSpArray, 50000, 5000, 2000, k, 10);
			SIMDSparsePerformance.matrixMultTest(sparsitiesExponential, denseSpArray, 100000, 5000, 2000, k, 10);
		}

		for(int k : threads) { // sparse dense MV
			SIMDSparsePerformance.matrixMultTest(sparsitiesLinear, denseSpArray, 10000000, 2000, 1, k, 100);
		}
	}

	public static void runSparseVectorizedElementWiseTests() {
		System.out.println("runSparseVectorizedElementWiseTests");

		for(int k : threads) { // sparse exp
			try {
				SIMDSparsePerformance.matrixExpTest(sparsitiesLinear, 100000, 10000, k, 30);
			} catch(Exception e) {
				e.printStackTrace();
			}

			try {
				SIMDSparsePerformance.matrixExpTest(sparsitiesExponential, 5000, 300000, k, 50);
			} catch(Exception e) {
				e.printStackTrace();
			}
		}

		for(int k : threads) { // sparse power
			for(double exponent : exponents) {
				try {
					SIMDSparsePerformance.matrixPowerTest(sparsitiesLinear, 100000, 10000, exponent, k, 50);
				} catch(Exception e) {
					e.printStackTrace();
				}

				try {
					SIMDSparsePerformance.matrixPowerTest(sparsitiesExponential, 5000, 300000, exponent, k, 90);
				} catch(Exception e) {
					e.printStackTrace();
				}
			}
		}

		for(int k : threads) { // sparse dense skip div MM
			try {
				SIMDSparsePerformance.matrixDivTest(sparsitiesLinear, denseSpArray, denseSpArray, 100000, 10000, 0, k, 30);
			} catch(Exception e) {
				e.printStackTrace();
			}

			try {
				SIMDSparsePerformance.matrixDivTest(sparsitiesExponential, denseSpArray, denseSpArray, 5000, 300000, 0, k, 70);
			} catch(Exception e) {
				e.printStackTrace();
			}
		}
	}

	public static void runFFMTests(String dmlPath) {
		System.out.println("runFFMTests");
		// Microbenchmark
		FFMPerformance.testMicrobenchmark("400-40000:10", "6000", 30);
		FFMPerformance.testMicrobenchmark("400-40000:10", "6000", 30);
		FFMPerformance.testMicrobenchmark("400-40000:10", "6000", 30);

		// JNI
		FFMPerformance.testNativeInvocation("1000", "1000", "1000", 1, 30, dmlPath);
		FFMPerformance.testNativeInvocation("8500", "8500", "8500", 1, 20, dmlPath);
		FFMPerformance.testNativeInvocation("16000", "16000", "16000", 1, 10, dmlPath);

		for(int k : threads) { // dense memory access
			FFMPerformance.testMemoryAccessMult("10000-70000#10000", "2000", "2000", k, 10, dmlPath); // varying lhs row
			FFMPerformance.testMemoryAccessMult("2000", "10000-70000#10000", "2000", k, 10, dmlPath); // varying lhs row
			FFMPerformance.testMemoryAccessMult("2000", "2000", "10000-70000#10000", k, 10, dmlPath); // varying lhs row
		}

		for(int k : threads) { // sparse memory access
			FFMPerformance.testMemoryAccessPower(sparsitiesLinear, 100000, 10000, 2.833, k, 30);
			FFMPerformance.testMemoryAccessPower(sparsitiesExponential, 5000, 300000, 2.833, k, 30);

		}
	}

	public static void sparseExpSingle() {
		try {
			SIMDSparsePerformance.matrixExpTest(sparsitiesLinear, 100000, 10000, 1, 30);
		} catch(Exception e) {
			e.printStackTrace();
		}
	}

	public static void sparseExpMulti() {
		try {
			SIMDSparsePerformance.matrixExpTest(sparsitiesLinear, 100000, 10000, Runtime.getRuntime().availableProcessors()/2, 30);
		} catch(Exception e) {
			e.printStackTrace();
		}
	}

	public static void sparseDenseSkipSingle() {
		try {
			SIMDSparsePerformance.matrixDivTest(sparsitiesLinear, denseSpArray, denseSpArray, 100000, 10000, 0, 1, 30);
		} catch(Exception e) {
			e.printStackTrace();
		}
	}

	public static void sparseDenseSkipMulti() {
		try {
			SIMDSparsePerformance.matrixDivTest(sparsitiesLinear, denseSpArray, denseSpArray, 100000, 10000, 0, Runtime.getRuntime().availableProcessors()/2, 30);
		} catch(Exception e) {
			e.printStackTrace();
		}
	}

	public static void sparsePowerMultiExponentFloat() {
		try {
			SIMDSparsePerformance.matrixPowerTest(sparsitiesLinear, 100000, 10000, 2.833, Runtime.getRuntime().availableProcessors(), 50);
		} catch(Exception e) {
			e.printStackTrace();
		}
	}

	public static void sparsePowerMultiExponentInt() {
		try {
			SIMDSparsePerformance.matrixPowerTest(sparsitiesLinear, 100000, 10000, 2.0, Runtime.getRuntime().availableProcessors(), 50);
		} catch(Exception e) {
			e.printStackTrace();
		}
	}

	public static void runAllDense(String dmlPath) {
		System.out.println("runAllDense");
		try {
			runDenseVectorizedMultTests(dmlPath);
			runSparseVectorizedMultTests(dmlPath);
		} catch(Exception e) {
			e.printStackTrace();
		}

		try {
			runDenseVectorizedElementWiseTests();
		} catch(Exception e) {
			e.printStackTrace();
		}
	}

	public static void runAllSparse(String dmlPath) {
		System.out.println("runAllSparse");

		try {
			runSparseVectorizedElementWiseTests();
		} catch(Exception e) {
			e.printStackTrace();
		}

		try {
			runSparseVectorizedMultTests(dmlPath);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void runAll(String dmlPath) {
		runFFMTests(dmlPath);
		runAllSparse(dmlPath);
	}
}
