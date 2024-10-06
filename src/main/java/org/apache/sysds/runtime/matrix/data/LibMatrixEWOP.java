package org.apache.sysds.runtime.matrix.data;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class LibMatrixEWOP {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    //TODO: Add documentation?
    //TODO: Add k = multi-threading?

    public static void exp(MatrixBlock a, MatrixBlock ret) {
        if(a.sparse) {
            expSparse(a, ret);
        } else {
            expDense(a, ret);
        }
    }

    public static void expSparse(MatrixBlock a, MatrixBlock ret) {
        //TODO
    }

    public static void expDense(MatrixBlock a, MatrixBlock ret) {
        double[] aVals = a.getDenseBlockValues();
        double[] retVals = ret.getDenseBlockValues();
        int len = aVals.length;
        DoubleVector aVec;
        int i = 0;
        int specLen = SPECIES.length();

        // Rest
        for (; i < len % specLen; i++) {
            retVals[i] = Math.exp(aVals[i]);
        }

        // Vectorized iteration
        for (; i <= SPECIES.loopBound(len); i += specLen) {
            aVec = DoubleVector.fromArray(SPECIES, aVals, i);
            aVec.lanewise(VectorOperators.EXP).intoArray(retVals, i);
        }
    }

    public static void power(MatrixBlock a, MatrixBlock ret, double exponent) {
        if(a.sparse) {
            powerSparse(a, ret, exponent);
        } else {
            powerDense(a, ret, exponent);
        }
    }

    public static void powerSparse(MatrixBlock a, MatrixBlock ret, double exponent) {
        //TODO
    }

    public static void powerDense(MatrixBlock a, MatrixBlock ret, double exponent) {
        double[] aVals = a.getDenseBlockValues(); // Input matrix
        double[] retVals = ret.getDenseBlockValues(); // Output matrix
        int len = aVals.length;

        DoubleVector aVec;

        // Vectorized loop for dense matrix
        int i = 0;
        int specLen = SPECIES.length();

        // Rest
        for (; i < len % specLen; i++) {
            retVals[i] = Math.exp(aVals[i]);
        }

        // Vectorized iteration
        for (; i <= SPECIES.loopBound(len); i += specLen) {
            aVec = DoubleVector.fromArray(SPECIES, aVals, i);
            aVec.lanewise(VectorOperators.POW, exponent).intoArray(retVals, i);
        }
    }


    public static void diff(MatrixBlock a, MatrixBlock b, MatrixBlock ret) {
        if(a.sparse || b.sparse) {
            diffSparse(a, b, ret);
        } else {
            diffDense(a, b, ret);
        }
    }

    private static void diffSparse(MatrixBlock a, MatrixBlock b, MatrixBlock ret) {
        //TODO
    }

    private static void diffDense(MatrixBlock a, MatrixBlock b, MatrixBlock ret) {
        double[] dataA = a.getDenseBlockValues();
        double[] dataB = b.getDenseBlockValues();
        double[] resultData = ret.getDenseBlockValues();
        int len = a.getNumRows() * a.getNumColumns();
        int specLen = SPECIES.length();
        int i = 0;
        DoubleVector aAsVec, bAsVec;

        // Rest
        for (; i < len % specLen; i++) {
            resultData[i] = dataA[i] - dataB[i];
        }

        // Vectorized iteration
        for (; i <= SPECIES.loopBound(len); i += specLen) {
            aAsVec = DoubleVector.fromArray(SPECIES, dataA, i);
            bAsVec = DoubleVector.fromArray(SPECIES, dataB, i);
            aAsVec.sub(bAsVec).intoArray(resultData, i);
        }
    }
}
