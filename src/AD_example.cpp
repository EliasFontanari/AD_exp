/**
 * \file AD_example.cpp
 * \brief Example of root finding for nonlinear systems with the Newton solver.
 */

#include <stdio.h>
#include <iostream>
#include "Newton_solver.hh"
#include <chrono>             // include chrono to measure time

auto start = std::chrono::high_resolution_clock::now();  // Start timing

// Eigen::Matrix<NewtonSolver<3, 2>::AD_N, 2, 1> my_func_ad(
//         const Eigen::Matrix<NewtonSolver<3, 2>::AD_N, 3, 1>& v)
// {
//     auto x = v[0], y = v[1], z = v[2];
//     Eigen::Matrix<NewtonSolver<3, 2>::AD_N, 2, 1> out;
//     out << x * x + sin(y),
//            x * y - exp(z) + z;
//     return out;
// }

// Eigen::Matrix<NewtonSolver<4, 4>::AD_N, 4, 1> Wood(
//         const Eigen::Matrix<NewtonSolver<4, 4>::AD_N, 4, 1>& v)
// {
//     auto x = v[0], y = v[1], z = v[2], w = v[3];
//     Eigen::Matrix<NewtonSolver<4, 4>::AD_N, 4, 1> out;
//     out << 10 * (y - pow(x,2)),
//            1-x,
//            sqrt(90) * (w - pow(z,2)),
//            1 - z;
//     return out;
// }

/**
 * @brief Helical Valley function, a common test problem for optimization algorithms. Every systems of equation has to be described by a function of this form, that is passes as argument to the solver.
 * @tparam Scalar The scalar type of the input and output vectors. The solver requires double and double AD scalar type, the type used by TinyAD.
 * @param v The input vector representing the point at which to evaluate the function.
 * @param out The output vector where the function value will be stored.
 */
template<typename Scalar>
void HelicalValley(
    const Eigen::Matrix<Scalar, 3, 1>& v,Eigen::Matrix<Scalar, 3, 1>& out)
{
    auto x = v[0], y = v[1], z = v[2];
    out << 10 * (z - 10 * atan2(y, x) / (2 * M_PI)),
           10 * (sqrt(x*x + y*y) - 1),
           z;
}

// template<typename Scalar>
// Eigen::Matrix<Scalar, 8, 1> ExtendedPowell(
//     const Eigen::Matrix<Scalar, 8, 1>& v,
//     Eigen::Matrix<Scalar, 8, 1>& out)
// {
//     // i = 1 (0-indexed: 0)
//     out[0] = v[0] + 10.0 * v[1];
//     out[1] = sqrt(5.0) * (v[2] - v[3]);
//     out[2] = (v[1] - 2.0 * v[2]) * (v[1] - 2.0 * v[2]);
//     out[3] = sqrt(10.0) * (v[0] - v[3]) * (v[0] - v[3]);

//     // i = 5 (0-indexed: 4)
//     out[4] = v[4] + 10.0 * v[5];
//     out[5] = sqrt(5.0) * (v[6] - v[7]);
//     out[6] = (v[5] - 2.0 * v[6]) * (v[5] - 2.0 * v[6]);
//     out[7] = sqrt(10.0) * (v[4] - v[7]) * (v[4] - v[7]);
//     return out;
// }

// template<typename Scalar>
// Eigen::Matrix<Scalar, 10, 1> DiscreteBoundaryValue(
//     const Eigen::Matrix<Scalar, 10, 1>& v,
//     Eigen::Matrix<Scalar, 10, 1>& out)
// {
//     constexpr int N = 10;
//     constexpr double h = 1.0 / (N + 1);

//     for (int i = 0; i < N; ++i)
//     {
//         double t = (i + 1) * h;  // t_i = i*h, 1-indexed so t1..t10

//         Scalar x_prev = (i == 0)     ? Scalar(0.0) : v[i - 1];  // x0 = 0
//         Scalar x_curr = v[i];
//         Scalar x_next = (i == N - 1) ? Scalar(0.0) : v[i + 1];  // x11 = 0

//         Scalar bracket = x_curr + t + 1.0;
//         out[i] = 2.0 * x_curr - x_prev - x_next + (h * h / 2.0) * bracket * bracket * bracket;
//     }
//     return out;
// }

int main() {
    NewtonSolver<3, 3> solver;
    Eigen::Vector<double, 3> guess;
    
    guess << -15,0.0,0.0; // Initial guess for the solution
    solver.set_initial_guess(guess); // Set the initial guess for the solver

    solver.solve(HelicalValley<decltype(solver)::AD_N>,HelicalValley<double>);
    if (solver.get_success()) {  
        std::cout << "Solver converged in " << solver.get_iterations() << " iterations." << std::endl;
        std::cout << "Solution: " << solver.get_solution().transpose() << std::endl;
    } else {
        std::cout << "Solver did not converge after " << solver.get_iterations() << " iterations." << std::endl;
    }
    
    auto stop = std::chrono::high_resolution_clock::now(); // Stop timing
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); // Calculate the duration in microseconds
    std::cout << "\n\n";
    std::cout << "Execution time: " << duration.count() << " μs" << std::endl;
    return 0;
}