// LU-Decomposition.cpp : This is a C++ implementation of LU decomposition algorithm.

#include <iostream>
#include <vector>


void LUDecompose(std::vector<std::vector<double>>& A) {

    int n = A.size();

    // Step 0: Initialize the matrix of Lower and Upper.
    std::vector<std::vector<double>> Lower(n, std::vector<double>(n,0));
    for (int i = 0; i < n; i++) {
        Lower[i][i] = 1;
    }
    
    std::vector<std::vector<double>> Upper(n, std::vector<double>(n, 0));

    // LU Decomposition body: Compute and fill the elements of Lower and Upper.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j < i) { // in this branch: compute Lower[i][j]
                double temp = 0;
                for (int x = 0; x <= j - 1; x++) {
                    temp += Lower[i][x] * Upper[x][j];
                }
                
                temp = A[i][j] - temp;
                Lower[i][j] = temp / Upper[j][j];
                
                //std::cout << "Lower[" << i << "][" << j << "] = " << Lower[i][j] << std::endl;
            }
            else { // j >= i, in this branch, compute Upper[i][j]
                double temp = 0;
                for (int x = 0; x <= j; x++) {
                    temp += Lower[i][x] * Upper[x][j];
                }
                temp -= Lower[i][i] * Upper[i][j];
                temp = A[i][j] - temp;
                Upper[i][j] = temp / Lower[i][i];

                //std::cout << "Upper[" << i << "][" << j << "] = " << Upper[i][j] << std::endl;
            }
        }
    }


    // Print matrix Lower
    std::cout << "Lower matrix: \n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j < n - 1) {
                std::cout << Lower[i][j] << ", ";
            }
            else {
                std::cout << Lower[i][j] << ", \n";
            }
        }
    }


    std::cout << std::endl;

    // Print matrix Upper
    std::cout << "Upper matrix:" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << "Line " << i << ": ";
        for (int j = 0; j < n; j++) {
            if (j < n - 1) {
                std::cout << Upper[i][j] << ", ";
            }
            else {
                std::cout << Upper[i][j] << "\n";
            }
        }
    }
}



int main()
{
    std::vector<std::vector<double>> A = { {2,3,2}, {1,3,2}, {3,4,1} };
    //std::vector<std::vector<double>> A = { {8,2,9}, {4,9,4}, {6,7,9} };

    LUDecompose(A);
    std::cout << "Computation complete!\n";
    return 0;
}

