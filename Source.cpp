#include <mpi.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include <random>
#include <algorithm>
#include <functional>
#include <cassert>
#include <limits>
#include <cmath>
using namespace std;
#define PI 3.14159265   

void initial_distribution(double* u, int N)
{
    for (int i = 1; i < N - 1; ++i) {
        u[i] = 1;
    }
    u[0] = 0.0;
    u[N - 1] = 0.0;
}

void swap(double** T0, double** T1)
{
    double* tmp;
    tmp = (*T0);
    //cout << T0[0][1] << endl;
    (*T0) = (*T1);
    (*T1) = tmp;
    //cout << T0[0][1] << endl;
}

void communication(const double sLeft, const double sRight, double& rLeft, double& rRight, const int my_rank, const int total_ranks, const int tag)
{
    MPI_Status status;

    if (my_rank < total_ranks - 1)
    {
        MPI_Send(&sRight, 1, MPI_DOUBLE, my_rank + 1, tag , MPI_COMM_WORLD);
    }
    if (my_rank > 0)
    {
        MPI_Recv(&rLeft, 1, MPI_DOUBLE, my_rank - 1, tag , MPI_COMM_WORLD, &status);
    }
    if (my_rank > 0)
    {
        MPI_Send(&sLeft, 1, MPI_DOUBLE, my_rank - 1, tag + 1, MPI_COMM_WORLD);
    }
    if (my_rank < total_ranks - 1)
    {
        MPI_Recv(&rRight, 1, MPI_DOUBLE, my_rank + 1, tag  + 1, MPI_COMM_WORLD, &status);
    }
   

}


void calculations(const double* T0, double* T1, const int size, const double left, const double right, const double dx,
    const double dt)
{
    for (int i = 0; i < size; ++i)
    {
        if (i == 0)
        {
            T1[i] = T0[i] + dt / (dx * dx) * (left - 2 * T0[i] + T0[i + 1]);
        }
        else if (i == size - 1)
        {
            T1[i] = T0[i] + dt / (dx * dx) * (T0[i - 1] - 2 * T0[i] + right);
        }
        else
        {
            T1[i] = T0[i] + dt / (dx * dx) * (T0[i - 1] - 2 * T0[i] + T0[i + 1]);
        }
    }
}

double* solve(const double* u, const int Nr, const int M, const double tau, const double h, const int my_rank, const int total_ranks)
{
    double* T0 = new double[Nr];
    double* T1 = new double[Nr];
    for (int i = 0; i < Nr; ++i)
    {
        T0[i] = u[my_rank * Nr + i];
        T1[i] = 0;
    }

    for (int i = 1; i < M; ++i)
    {
        double sLeft = T0[0], sRight = T0[Nr - 1];
        double rLeft = 0.0, rRight = 0.0;

        communication(sLeft, sRight, rLeft, rRight, my_rank, total_ranks, i);
        calculations(T0, T1, Nr, rLeft, rRight, h, tau);
        swap(&T0, &T1);
    }
    
    return T0;
}

void solveVerify(double* u, const int N,  const double l, const float T) {
    double eps = 0.0001;
    
    for (int i = 0; i < N; ++i) {
        int m = 0;
        double an = 1.0;
        u[i] = 0;
        while (fabs(an) >= eps) {
            an = exp(-1 * (PI * PI) * (2 * m + 1) * (2 * m + 1) * T / (l * l)) / (2 * m + 1) * sin(PI * (2 * m + 1) * i * 0.1 / l);
            u[i] += an;
            m++;
            
        }
        u[i] *= (4 * 1 / PI);
        //cout << u[i] << endl;
    }
}

bool verifyResult(double* u, const int N, const double l, const float T,
    const double accuracy = 0.1) {
    double* uCorr = new double[11];
    solveVerify(uCorr, 11, l, T);

    for (int i = 0; i < 11; ++i) {
        if (fabs(uCorr[i] - (u[(i * N / 10)])) > accuracy) {
            cout <<i<<"   "<< fabs(uCorr[i] - (u[(i * N / 10) - 1])) << endl;
            
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {

    int XSize = 10001;
    int total_ranks = 0;
    int my_rank = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &total_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int* nums = nullptr;
    int* displs = nullptr;
    double* res = nullptr;
    double* localRes = nullptr;
    double timeSum = 0.0;
    double l = 1.0;
    double h = 1.0 / (XSize - 1);
    float tau = 0.00000002;
    float T = 0.001;
    int M = (T / tau) + 1;
    double* u = new double[XSize];
    initial_distribution(u, XSize);

    int Nr = XSize / total_ranks;
    if (my_rank < XSize % total_ranks)
    {
        Nr = Nr + 1;
    }

    auto startTime = MPI_Wtime();

    localRes = solve(u, Nr, M, tau, h, my_rank, total_ranks);
    /*if (rank == 0) {
        for (int i = 0; i < Nr; ++i)
            cout << i << "=" << localRes[i] << endl;
    }*/

    if (my_rank == 0)
    {
       
        nums = (int*)malloc(sizeof(int) * total_ranks);
    }

    MPI_Gather(&Nr, 1, MPI_INT, nums, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //cout << nums;
    if (my_rank == 0)
    {
        displs = (int*)malloc(total_ranks * sizeof(int));
        displs[0] = 0;
        for (int i = 1; i < total_ranks; ++i) {
            displs[i] = displs[i - 1] + nums[i - 1];
        }
        res = (double*)malloc(XSize * sizeof(double));
    }
    
    MPI_Gatherv(localRes, Nr, MPI_DOUBLE, res, nums, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   /* if (rank == 0) {
        for (int i = 0; i < rowSize; ++i)
            cout <<i<<"="<< res[i] << endl;
    }*/
    auto endTime = MPI_Wtime();
    double procTime = (endTime - startTime);

    MPI_Reduce(&procTime, &timeSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0)
    {
        cout << "Threads: " << total_ranks << endl
            << "tau= " << tau << endl
            << "h^2= " << h * h << endl
            << "Time T=" << T << endl
            << "step through space " << XSize << endl
            << "Iterations in time (amount of dt): " << M << endl
            << "Sec: " << timeSum / total_ranks << endl;
       
        if (verifyResult(res, XSize-1, l, T)==false) {
            cout << "Verification failed!" << std::endl;
        }
        else {
            cout << "Verification successful!!!!!" <<endl;
        }

    }
    MPI_Finalize();
    return 0;
}