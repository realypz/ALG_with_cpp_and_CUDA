// Kmeans.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <fstream>

//void* operator new(size_t size) {
//    std::cout << "Allocating " << size << " memory.\n";
//    return malloc(size);
//}


template<typename _T>
class Matrix
{
public:
    _T* _p_elems;
    _T** _p_rows;
    int nj, ni;

public:
    Matrix(int const& nj, int const& ni) : nj(nj), ni(ni)
    {
        _p_elems = new _T[nj * ni];  // j: 行标号，i: 列标号
        _p_rows = new _T * [nj];
        for (int j = 0; j != nj; ++j)
        {
            _p_rows[j] = _p_elems + ni * j;
        }
        // std::cout << "matrix created!\n";
    }

    ~Matrix() {
        delete _p_elems;
        delete _p_rows;
    }

    void InitializeData(_T* arr) {
        /* Copy memory to the new dataset */
        for (int i = 0; i < nj * ni; i++) {
            *(this->_p_elems + i) = *(arr + i);
        }
    }


    /* Methods */
    _T sum() {
        _T acc = 0.0;
        int n_elements = this->nj * this->ni;

        for (int idx = 0; idx < n_elements; idx++) {
            acc = acc + *(this->_p_elems + idx);
        }
        return acc;
    }


    void zeros() {
        for (int i = 0; i < nj * ni; i++) {
            *(this->_p_elems + i) = 0.0;
        }
    }

    int CountElements() {
        return this->nj * this->ni;
    }








    /* Operators */
    _T const& operator()(int const& j, int const& i) const  
    {
        return _p_rows[j][i];
    }

    _T& operator()(int const& j, int const& i)
        // & means return by reference
    {
        return _p_rows[j][i];
    }

    _T const* operator[](int const& j) const
    {
        return _p_rows[j];
    }

    _T* operator[](int const& j)
    {
        return _p_rows[j];
    }


    _T& Get(const int& j, const int& i) {
        return _p_rows[j][i];
    }

    _T Get(const int& j, const int& i) const {
        return _p_rows[j][i];
    }


    Matrix<_T> operator+(const Matrix<_T>& mat0) {

        try {
            if (mat0.nj != this->nj || mat0.ni != this->ni) {
                throw "Cannot be added since two matrixs are does not have the same shape.";
            }

            Matrix<_T> sum_mat(this->nj, this->ni);

            for (int j = 0; j < this->nj; j++) {
                for (int i = 0; i < this->ni; i++) {
                    sum_mat(j, i) = this->_p_rows[j][i] + mat0(j, i);
                }
            }

            return sum_mat;
        }
        catch (const char* msg) {
            std::cerr << msg << std::endl;
        }

    }


};


template<typename _T>
class Kmeans {
public:
    int D;  // ndim
    int N;
    int K;
    int* labels;

    Matrix<_T>* data; // N * D
    Matrix<_T>* distance;  // N * K
    Matrix<_T>* centers;  // K * D


public:
    Kmeans(Matrix<_T>* data, int N, int D, int K)
        : data(data), N(N), D(D), K(K) {

        labels = new int[N];
        for (int i = 0; i < N; i++) {
            labels[i] = -1;
        }

        distance = new Matrix<_T>(N, K);
        centers = new Matrix<_T>(K, D);
        
    }


    ~Kmeans() {
        delete this->distance;
        delete this->centers;
        delete this->data;
        delete this->labels;
    }


    void Cluster(int steps) {
        this->InitializeCenter();
        for (int step = 0; step < steps; step++) {
            this->ComputeDistance();
            this->UpdateLabel();
            this->UpdateCenters();
        }
    }


    void InitializeCenter() {
        /* https://en.cppreference.com/w/cpp/algorithm/random_shuffle */

        std::vector<int> idx_shuffle;
        idx_shuffle.reserve(this->N);

        for (int i = 0; i < this->N; i++) {
            idx_shuffle.push_back(i);
        }

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(idx_shuffle.begin(), idx_shuffle.end(), g);

        int idx = 0;
        for (int k = 0; k < this->K; k++) {
            idx = idx_shuffle[k];
            for (int d = 0; d < this->D; d++) {
                this->centers->Get(k, d) = this->data->Get(idx, d);
            }
        }
    }


    void ComputeDistance() {
        _T dis = 0.0;
        _T err = 0.0;
        for (int i = 0; i < this->N; i++) {
            for (int k = 0; k < this->K; k++) {
                dis = 0.0;
                for (int d = 0; d < this->D; d++) {
                    err = this->data->Get(i,d) - this->centers->Get(k,d);
                    dis = dis + err * err;
                }
                this->distance->Get(i,k) = dis;
            }

        }
    }

    void UpdateLabel() {
        int label_i = 0;
        _T least_distance_i = 0.0;
        _T distance_ik = 0.0;

        for (int i = 0; i < this->N; i++) {

            label_i = 0;
            least_distance_i = 1.7e300;
            
            for (int k = 0; k < this->K; k++) {
                distance_ik = this->distance->Get(i, k);
                
                if (distance_ik < least_distance_i) {
                    label_i = k; 
                    least_distance_i = distance_ik;
                }
                    
            }

            this->labels[i] = label_i;
        }
    }


    void UpdateCenters() {

        Matrix<_T>* new_centers = new Matrix<_T>(this->K, this->D);
        new_centers->zeros();

        int* count = new int[this->K];
        for (int k = 0; k < this->K; k++) {
            count[k] = 0;
        }

        // sum all dim
        _T _tmp = 0.0;
        for (int i = 0; i < this->N; i++) {
            int label_i = this->labels[i];
            count[label_i] ++;
            for (int d = 0; d < this->D; d++) {
                _tmp = new_centers->Get(label_i, d) + this->data->Get(i, d);
                new_centers->Get(label_i, d) = _tmp;
            }
            
        }

        int count_k = 0;
        // get average
        for (int k = 0; k < this->K; k++) {
            count_k = count[k];
            for (int d = 0; d < this->D; d++) {
                this->centers->Get(k, d) = new_centers->Get(k, d) / count_k;
            }
        }

        delete new_centers;
    }

};






int main()
{

    /* Read file */
    std::ifstream file;
    file.open("./dataset/s1.txt");

    if (!file) {
        std::cout << "Unable to open the file.\n";
        return 0;
    }

    int N = 5000;
    int D = 2;
    double* arr = new double[N * D];
    double x;

    double* ptr = arr;

    /* Copy the data from file to memory. */
    while (file >> x) {
        *ptr = x / 100.0;  // Scale down the input data, to avoid overflow.
        ptr++;
    }
    file.close();


    auto data_matrix = new Matrix<double>(N, D);
    data_matrix->InitializeData(arr);

    int K = 15;
    int steps = 30;

    auto kmeans_obj = new Kmeans<double>(data_matrix, N, D, K);

    /* Do the clustering computation. */
    kmeans_obj->Cluster(steps);

    /* Write the clustering result labels to a file. */
    std::ofstream result_file("./dataset/s1_result.txt");
    if (result_file.is_open())
    {
        for (int count = 0; count < N; count++) {
            result_file << kmeans_obj->labels[count] << std::endl;
        }
        result_file.close();
    }

    return 0;
}

