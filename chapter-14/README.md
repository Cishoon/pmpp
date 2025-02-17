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

**COO**
We need `z` integers for `rowidx` matrix, `z` integers for `colidx` matrix and `z` integers for `value` matrix, so `3z` integers to represent the matrix. 

**CSR**

We need `z` integers for `colidx`arraymatrix, `z` integers for `value` array and we need to store `m+1` pointers in `rowptrs` array. So the total of `z + z + m + 1` integers. 

**ELL**

We don't have enough information to fully estimate how many itegers we do need. In `ELL` we pad the rows to match the len to the longest row. Here we lack the infromation how many integers are in the longest one. 

Assuming that all of the rows would be the same (no padding), we would need `z` integers for `colidx` array and `z` integers for `value` array - so `2z` integers in total. In practice we will have `2z + padding`. 

**JDS**

Here we also lack some crucial information, namely how many non zero numbers are in the row with the most non-zero numbers. We need this information to know how many integers we need to store in the `iterptr` array. 

Assuming that we have the same number of non-zero numbers in each row, we would have `z/m + 1` integers row in each row. We would need `z/m + 1` integers for `iterptr`, `z` integers for `colidx` array, and `z` integers for `value` array. So `z + z + z/m + 1` integers in total. 


### Exercise 3

**Implement the code to convert from COO to CSR using fundamental parallel computing primitives, including histogram and prefix sum.**

### Exercise 4

**Implement the host code for producing the hybrid ELL-COO format and using it to perform SpMV. Launch the ELL kernel to execute on the device, and compute the contributions of the COO elements on the host.**

### Exercise 5

**Implement a kernel that performs parallel SpMV using a matrix stored in the JDS format.**