package org.apache.sysds.performance.matrix;

import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Power;
import org.apache.sysds.runtime.functionobjects.Power2;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell;
import org.apache.sysds.runtime.matrix.data.LibMatrixDiffExpPow2;
import org.apache.sysds.runtime.matrix.data.LibMatrixEWOP;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

public class MatrixElementWiseSIMDPerformance {
    private static final double EPSILON = 1E-10;

    public static void diffExpPowerTest() {
        MatrixBlock A = MatrixBlock.randOperations(5, 5, 1, 0, 1, "uniform", 7);
        MatrixBlock B = MatrixBlock.randOperations(A.getNumRows(), A.getNumColumns(), 1, 0, 1, "uniform", 8);
        MatrixBlock minusC = new MatrixBlock(A.getNumRows(), A.getNumColumns(), false);
        MatrixBlock minusCSIMD = MatrixBlock.randOperations(A.getNumRows(), A.getNumColumns(), 1, 0, 1, "uniform", 9);
        MatrixBlock powerC2 = new MatrixBlock(A.getNumRows(), A.getNumColumns(), false);
        MatrixBlock powerC = new MatrixBlock(A.getNumRows(), A.getNumColumns(), false);
        MatrixBlock expC, expC2;
        expC2 = MatrixBlock.randOperations(A.getNumRows(), A.getNumColumns(), 1, 0, 1, "uniform", 9);

        double exponent = 3.0;
        BinaryOperator powerOp = new BinaryOperator(Power.getPowerFnObject());
        RightScalarOperator powerOpK = new RightScalarOperator(Power.getPowerFnObject(), exponent);
        UnaryOperator expOperator = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.EXP));

        LibMatrixBincell.bincellOp(A, B, minusC, new BinaryOperator(Minus.getMinusFnObject()));
        LibMatrixBincell.uncellOp(A, powerC2, new UnaryOperator(Power2.getPower2FnObject()));
        expC = A.unaryOperations(expOperator, new MatrixBlock());
        powerC = A.scalarOperations(powerOpK, new MatrixBlock());

        LibMatrixDiffExpPow2.diff(A, B, minusCSIMD);
        LibMatrixEWOP.exp(A, expC2);
        System.out.println(compareResults(expC, expC2, minusC.getNumRows(), minusC.getNumColumns()));

        System.out.println("A: " + A);
        System.out.println("B: " + B);
        System.out.println("C Minus: " + minusC);
        System.out.println("C Power2: " + powerC2);
        System.out.println("C Power: " + powerC);
        System.out.println("C Exp: " + expC);
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

    public static void matrixDiffTest(double sparsity1, double sparsity2, int rows, int cols, int k, int warmupIterations) {
        //TODO
    }

    public static void matrixPowerTest(double sparsity, int rows, int cols, double exponent, int k, int warmupIterations) {
        //TODO
    }

    public static void matrixExpTest(double sparsity, int rows, int cols, int k, int warmupIterations) {
        //TODO
    }

}
