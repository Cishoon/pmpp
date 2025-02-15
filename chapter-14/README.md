# Chapter 14

## Code

## Exercises

### Exercise 1

**Consider the following sparse matrix:**

```
1 0 7 0
0 0 8 0
0 4 3 0
2 0 0 1
```

**Represent it in each of the following formats: (1) COO, (2) CSR, (3) ELL, and (4) JDS.**

### Exercise 2

**Given a sparse matrix of integers with m rows, n columns, and z nonzeros, how many integers are needed to represent the matrix in (1) COO, (2) CSR, (3) ELL, and (4) JDS? If the information that is provided is not enough to allow an answer, indicate what information is missing.**

### Exercise 3

**Implement the code to convert from COO to CSR using fundamental parallel computing primitives, including histogram and prefix sum.**

### Exercise 4

**Implement the host code for producing the hybrid ELL-COO format and using it to perform SpMV. Launch the ELL kernel to execute on the device, and compute the contributions of the COO elements on the host.**

### Exercise 5

**Implement a kernel that performs parallel SpMV using a matrix stored in the JDS format.**