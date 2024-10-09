package org.apache.sysds.runtime.matrix.data;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.apache.sysds.runtime.data.SparseBlock;

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

    private static void expSparse(MatrixBlock a, MatrixBlock ret) {
        //TODO
    }

    private static void expDense(MatrixBlock a, MatrixBlock ret) {
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
        for (; i < SPECIES.loopBound(len); i += specLen) {
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

    private static void powerSparse(MatrixBlock a, MatrixBlock ret, double exponent) {
//        ret.allocateSparseRowsBlock();
//
//        SparseBlock blockA = a.getSparseBlock();
//        SparseBlock blockRet = a.getSparseBlock();
//
//        int ru = a.getNumRows();
//        int specLen = SPECIES.length();
//
//        // Iterate over all rows
//        for (int r = 0; r < ru; r++) {
//            if (blockA.isEmpty(r)) continue;
//
//            int apos = blockA.pos(r);
//            int alen = blockA.size(r);
//            int[] aix = blockA.indexes(r);
//            double[] avals = blockA.values(r);
//            DoubleVector aVec;
//
//            int i = 0;
//
//            // Rest
//            for (; i < alen % specLen; i++) {
//                blockRet[i] = Math.pow(aVals[i], exponent);
//            }
//
//            // Vectorized iteration
//            for (; i <= SPECIES.loopBound(len); i += specLen) {
//                aVec = DoubleVector.fromArray(SPECIES, aVals, i);
//                aVec.lanewise(VectorOperators.POW, exponent).intoArray(retVals, i);
//            }
//
//            double val = values[pos + j];
//            int col = indexes[pos + j];
//
//            double result = Math.pow(val, exponent);
//
//            ret.setValueSparseUnsafe(r, col, result);
//        }
//
//        // Set the number of non-zeros in the result matrix
//        ret.setNonZeros(a.getNonZeros());
    }

    private static void powerDense(MatrixBlock a, MatrixBlock ret, double exponent) {
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
        for (; i < SPECIES.loopBound(len); i += specLen) {
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
        for (; i < SPECIES.loopBound(len); i += specLen) {
            aAsVec = DoubleVector.fromArray(SPECIES, dataA, i);
            bAsVec = DoubleVector.fromArray(SPECIES, dataB, i);
            aAsVec.sub(bAsVec).intoArray(resultData, i);
        }
    }
}
