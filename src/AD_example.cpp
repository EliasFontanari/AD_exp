#include <stdio.h>
#include <iostream>
#include "Newton_solver.hh"
#include <chrono>             // include chrono to measure time

auto start = std::chrono::high_resolution_clock::now();  // Start timing

Eigen::Matrix<double, 2, 1> my_func(const Eigen::Matrix<double, 3, 1>& v)
{
    double x = v[0], y = v[1], z = v[2];
    return { x*x + sin(y),
             x*y - exp(z) + z };
}
Eigen::Matrix<NewtonSolver<3, 2>::AD_N, 2, 1> my_func_ad(
        const Eigen::Matrix<NewtonSolver<3, 2>::AD_N, 3, 1>& v)
{
    auto x = v[0], y = v[1], z = v[2];
    Eigen::Matrix<NewtonSolver<3, 2>::AD_N, 2, 1> out;
    out << x * x + sin(y),
           x * y - exp(z) + z;
    return out;
}

Eigen::Matrix<NewtonSolver<4, 4>::AD_N, 4, 1> Wood(
        const Eigen::Matrix<NewtonSolver<4, 4>::AD_N, 4, 1>& v)
{
    auto x = v[0], y = v[1], z = v[2], w = v[3];
    Eigen::Matrix<NewtonSolver<4, 4>::AD_N, 4, 1> out;
    out << 10 * (y - pow(x,2)),
           1-x,
           sqrt(90) * (w - pow(z,2)),
           1 - z;
    return out;
}

Eigen::Matrix<NewtonSolver<3, 3>::AD_N, 3, 1> HelicalValley(
        const Eigen::Matrix<NewtonSolver<3, 3>::AD_N, 3, 1>& v)
{
    auto x = v[0], y = v[1], z = v[2];
    Eigen::Matrix<NewtonSolver<3, 3>::AD_N, 3, 1> out;
    out << 10 * (z - 10*atan2(y,x)/(2*M_PI)),
           10*(sqrt(x*x+y*y) - 1),
           z;
    return out;
}

int main() {
    NewtonSolver<3, 3> solver;
    Eigen::Vector<double, 3> guess;
    
    guess << 310.0, -8.0, 5.;
    solver.set_initial_guess(guess);


    // solver.print_initial_guess();
    // std::cout << "Initial settings: " << solver.get_settings().max_iterations << ", " << solver.get_settings().tol << std::endl;
    
    // solver.Jacobian(solver.initial_guess, my_func_ad);
    // if (solver.J)
    //     std::cout << "Jacobian at initial guess:\n" << *solver.J << std::endl;

    // std::cout << "Evaluating function at initial guess:" << solver.initial_guess << std::endl;
    solver.solve(HelicalValley);
    if (solver.get_success()) {  
        std::cout << "Solver converged in " << solver.get_iterations() << " iterations." << std::endl;
        std::cout << "Solution: " << solver.get_solution().transpose() << std::endl;
        // std::cout << "Function evaluation at transpose " << solver.F0.transpose() << std::endl;

    } else {
        std::cout << "Solver did not converge after " << solver.get_iterations() << " iterations." << std::endl;
    }
    // auto G = my_func_ad(guess);
    // solver.assign_F0(G);
    // std::cout << solver.F0 << std::endl;
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "\n\n";
    std::cout << "Execution time: " << duration.count() << " μs" << std::endl;
    return 0;
}