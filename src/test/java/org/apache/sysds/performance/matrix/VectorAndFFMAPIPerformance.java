package org.apache.sysds.performance.matrix;

import static org.apache.sysds.performance.matrix.MatrixSIMDPerformance.matrixMultTest;

public class VectorAndFFMAPIPerformance {

    private static double denseSparsity = 0.8;
    private static double[] sparseSparsities = {0.3, 0.1, 0.01};
    private static int[] threads = {1, Runtime.getRuntime().availableProcessors()}; // Get maximum amount of threads //TODO: test on servers
    
    public static void runMatrixMultTests(String dmlPath) {
        // DenseDense Small tests (matrix sizes <= 5000)
        String sizesSm = "1000-5000#1000";
        int warmupRunsSm = 80;
        matrixMultTest(1.0, 1.0, sizesSm, sizesSm, sizesSm, 32, 80, dmlPath);
        matrixMultTest(1.0, 0.7, sizesSm, sizesSm, sizesSm, 32, 80, dmlPath);
        matrixMultTest(1.0, 0.5, sizesSm, sizesSm, sizesSm, 32, 80, dmlPath);
        matrixMultTest(0.7, 0.9, sizesSm, sizesSm, sizesSm, 32, 80, dmlPath);
        matrixMultTest(0.7, 0.5, sizesSm, sizesSm, sizesSm, 32, 80, dmlPath);

        // DenseDense Medium tests (matrix sizes >= 5000 and <= 25000)
        String sizesMd = "5000-25000#5000";
        int warmupRunsMd = 50;
        matrixMultTest(1.0, 1.0, sizesMd, sizesMd, sizesMd, 32, 50, dmlPath);
        matrixMultTest(1.0, 0.7, sizesMd, sizesMd, sizesMd, 32, 50, dmlPath);
        matrixMultTest(1.0, 0.5, sizesMd, sizesMd, sizesMd, 32, 50, dmlPath);

        // #################################################################################################
        // DenseSparse Small tests (matrix sizes <= 5000)
        sizesSm = "1000-5000#1000";
        warmupRunsSm = 50;
        matrixMultTest(1.0, 1.0, sizesSm, sizesSm, sizesSm, 32, 100, dmlPath);
        matrixMultTest(1.0, 0.7, sizesSm, sizesSm, sizesSm, 32, 80, dmlPath);
        matrixMultTest(1.0, 0.5, sizesSm, sizesSm, sizesSm, 32, 80, dmlPath);
        matrixMultTest(0.7, 0.9, sizesSm, sizesSm, sizesSm, 32, 80, dmlPath);
        matrixMultTest(0.7, 0.5, sizesSm, sizesSm, sizesSm, 32, 80, dmlPath);

        // DenseSparse Medium tests (matrix sizes >= 5000 and <= 25000)
        sizesMd = "5000-25000#5000";
        warmupRunsMd = 50;
        matrixMultTest(1.0, 1.0, sizesMd, sizesMd, sizesMd, 32, 50, dmlPath);
        matrixMultTest(1.0, 0.7, sizesMd, sizesMd, sizesMd, 32, 50, dmlPath);
        matrixMultTest(1.0, 0.5, sizesMd, sizesMd, sizesMd, 32, 50, dmlPath);
        matrixMultTest(0.7, 0.9, sizesMd, sizesMd, sizesMd, 32, 50, dmlPath);
        matrixMultTest(0.7, 0.5, sizesMd, sizesMd, sizesMd, 32, 50, dmlPath);
    }

    public static void runMatrixDivTests() {
        //TODO
    }

    public static void runMatrixPowerTests() {
        //TODO
    }

    public static void runMatrixExpTests() {
        //TODO
    }

    public static void runFFMTests() {
        //TODO
    }
}
