// LU-Decomposition.cpp : This is a C++ implementation of LU decomposition algorithm.

#include <iostream>
#include <vector>
#include <string>


void PrintMatrix(std::vector<std::vector<double>>& M, std::string name);

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


    PrintMatrix(Lower, "Lower matrix");

    // Print matrix Upper
    PrintMatrix(Upper, "Upper matrix");
}



void LUDecompose_leftlooking_v2(std::vector<std::vector<double>> &A) {
    // reference https://intra.ece.ucr.edu/~stan/papers/tvlsi_gpu_lu14.pdf, Algorithm 1

    int n = A.size();
    auto As = A; // copy matrix A to As. The following for loop will update As.

    for (int j = 0; j <= n - 1; j++) {
        for (int k = 0; k <= j - 1; k++) {
            if (As[k][j] != 0) {
                for (int i = k + 1; i <= n - 1; i++) {
                    if (As[i][k] != 0) {
                        //printf("Update As[%d][%d], k=%d\n", i, j, k);
                        As[i][j] = As[i][j] - As[i][k] * As[k][j];
                    }
                }
            }
        }

        //std::cout << "---------------\n";

        for (int i = j + 1; i <= n - 1; i++) {
            //printf("Update As[%d][%d]\n", i, j);
            As[i][j] = As[i][j] / As[j][j];
        }

        //std::cout << "The end of " << j << "-th iteration.\n";
        //PrintMatrix(As, "As");
        //std::cout << "\n";
    }

    //The lower part of As is L, and the upper part of As is U.
    PrintMatrix(As, "As");
}


void LUDecomposition_v3(std::vector<std::vector<double>>& A) {

    // kji forms,  https://courses.engr.illinois.edu/cs554/fa2015/notes/06_lu.pdf, page 9.

    int n = A.size();
    auto As = A; // copy matrix A to As. The following for loop will update As.

    for (int k = 0; k <= n - 2; k++) {
        //std::cout << "k = " << k << std::endl;
        for (int i = k + 1; i <= n - 1; i++) {
            //printf("operate on As[%d][%d]\n", i, k);
            As[i][k] = As[i][k] / As[k][k];
        }
        //printf("==========\n");
        for (int j = k + 1; j <= n - 1; j++) {
            for (int i = k + 1; i <= n - 1; i++) {
                //printf("operate on As[%d][%d]\n", i, j);
                As[i][j] = As[i][j] - As[i][k] * As[k][j];
            }
        }
        //std::cout << "\n";
    }

    PrintMatrix(As, "As");
}









int main()
{
    std::vector<std::vector<double>> A = { {2,3,2,1.4}, {1,3,2,-0.7}, {3,-3,4,1}, {-3.2, 5.3, 4.5, 0.3} };
    //std::vector<std::vector<double>> A = { {8,2,9}, {4,9,4}, {6,7,9} };

    std::cout << "--------------LUDecompose--------------" << std::endl;
    LUDecompose(A);

    std::cout << "--------------LUDecompose_leftlooking_v2--------------" << std::endl;
    LUDecompose_leftlooking_v2(A);

    std::cout << "---------------LUDecomposition_v3-------------" << std::endl;
    LUDecomposition_v3(A);
    
    return 0;
}


void PrintMatrix(std::vector<std::vector<double>>& M, std::string name)
{
    int n = M.size();

    // Print matrix Lower
    std::cout << name << " : \n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j < n - 1) {
                std::cout << M[i][j] << ", ";
            }
            else {
                std::cout << M[i][j] << ", \n";
            }
        }
    }
    std::cout << std::endl;
}

