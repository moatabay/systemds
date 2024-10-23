package org.apache.sysds.performance.matrix;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.Arrays;
import java.util.Optional;

public class FFMPerformance {
    /* Matthias:
    first, micro benchmarking of allocations and then data access speed in
    elementwise and matmult operations. Yes, prototype-level or leaving it
    out is fine, if that way you complete the vector API in very good quality.

    Me:
    I looked through the code and I didn't see any spots where off-heap memory
    is allocated or accessed in the context of element-wise or matmult operations.
    I only saw on-heap memory allocations and accesses.

    I just wanted to make sure: The point is to compare the performance between
    the Java Foreign Memory API and the JNI (or ByteBuffer?) in terms of allocation
    and data access (of matrix blocks) from the off-heap memory, right?

    Matthias:
    there are multiple dimensions
    * compare data transfer for native BLAS operations via JNI with an
    alternative implementation with the Foreign memory API
    * implement our Java matrix multiplication and elementwise operations to
    work directly with the Foreign memory API.

    Me:
    As far as I understood, I need to implement three "kinds" of performance tests for the Foreign Memory API, right?

    Microbenchmark of allocations: For example the allocation of a 5000x5000 matrix in the Foreign Memory API versus ByteBuffers (or JNI?).
    Of course I will do it with multiple size.
    The invocation of native BLAS operations (for example "dmmdd") to compare the speed between JNI and the Foreign Memory API.
    And I still didn't really understand this part: "implement our Java matrix multiplication and elementwise operations to
    work directly with the Foreign memory API." because the Foreign Memory API is only there to invoke native functions and allocate off-heap memory.
    So I guess I didn't really understand how the Foreign Memory API can work with matrix multiplication (apart from calling LibMatrixNative)
    and the elementwise operations.

    Matthias:
    ad 3: you could change these operations to get every value over the
    foreign memory API (similar to using Unsafe), right? If so, we would
    want to quantify this overhead.
     */

    public static void test() {
        int m = 2; // Rows of A (resulting matrix C)
        int n = 2; // Columns of B (resulting matrix C)
        int k = 3; // Columns of A (must match rows of B)

        // Example matrix data for A (2x3) and B (3x2)
        double[] matrixAData = {
                1.0, 2.0, 3.0,  // Row 1
                4.0, 5.0, 6.0   // Row 2
        };

        double[] matrixBData = {
                7.0, 8.0,       // Row 1
                9.0, 10.0,     // Row 2
                11.0, 12.0     // Row 3
        };

        // Allocate native memory for matrices A, B, and C
        try (Arena arena = Arena.ofConfined()) {
            // Allocate memory segments for A, B, and C
            MemorySegment a = arena.allocateArray(ValueLayout.JAVA_DOUBLE, matrixAData);
            MemorySegment b = arena.allocateArray(ValueLayout.JAVA_DOUBLE, matrixBData);
            MemorySegment c = arena.allocateArray(ValueLayout.JAVA_DOUBLE, new double[m * n]); // C will be 2x2

            // Leading dimensions
            int lda = k; // Leading dimension for A (number of rows of A)
            int ldb = k; // Leading dimension for B (number of rows of B)
            int ldc = n; // Leading dimension for C (number of rows of C)

            double alpha = 1.0; // Scalar for A * B
            double beta = 0.0;  // Scalar for C

            Linker linker = Linker.nativeLinker();
            SymbolLookup lookup = linker.defaultLookup();

            Optional<MethodHandle> addMethodHandle = lookup.find("dmmdd")
                    .map(symbol -> linker.downcallHandle(symbol, FunctionDescriptor.ofVoid(
                            ValueLayout.JAVA_INT,  // Layout for A
                            ValueLayout.JAVA_INT,  // Layout for B
                            ValueLayout.JAVA_INT,  // M
                            ValueLayout.JAVA_INT,  // N
                            ValueLayout.JAVA_INT,  // K
                            ValueLayout.JAVA_DOUBLE, // Alpha
                            ValueLayout.ADDRESS,    // A
                            ValueLayout.JAVA_INT,   // LDA
                            ValueLayout.ADDRESS,    // B
                            ValueLayout.JAVA_INT,   // LDB
                            ValueLayout.JAVA_DOUBLE, // Beta
                            ValueLayout.ADDRESS,    // C
                            ValueLayout.JAVA_INT     // LDC
                    )));

            if(addMethodHandle.isPresent()) {
                addMethodHandle.get().invoke(0, 0, m, n, k, alpha, a.address(),
                        lda,
                        b.address(),
                        ldb,
                        beta,
                        c.address(),
                        ldc);
            } else {
                System.out.println("Didn't find lookup");
            }

            // Read result from native memory
            double[] result = c.toArray(ValueLayout.JAVA_DOUBLE);
            // Process the result as needed
            System.out.println("Result Matrix C:");
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    System.out.print(result[i * n + j] + " ");
                }
                System.out.println();
            }
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }
}
