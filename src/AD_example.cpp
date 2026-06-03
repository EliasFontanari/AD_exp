// #include <TinyAD/Scalar.hh>
#include <stdio.h>
#include <iostream>
#include "Newton_solver.hh"


// // Choose autodiff scalar type for 3 variables
// using ADouble = TinyAD::Double<3>;

// int main() {
//     // Init a 3D vector of active variables and a 3D vector of passive variables
//     Eigen::Matrix<double, 3, 1> guess;
//     guess << 1.0, 2.0, 0.5;
//     solver.set_initial_guess(guess);
// //     // simple test: try to write [x^2 +]
// //     // Eigen::Vector3<ADouble> x = ADouble::make_active({0.0, -1.0, 1.0});
//     auto x_ad = solver.initial_guess_active();
//     solver.Jacobian(x_ad, my_func_ad);
//     if (solver.J)
//         std::cout << "Jacobian at initial guess:\n" << *solver.J << std::endl;

//     // Eigen::Vector3<ADouble> xy = x+y;

//     // // Compute angle using Eigen functions and retrieve gradient and Hessian w.r.t. x
//     // ADouble angle = acos(x.dot(y) / (x.norm() * y.norm()));
//     // Eigen::Vector3d g = angle.grad;
//     // Eigen::Matrix3d H = angle.Hess;

//     TinyAD::Double<3> z = x+y;
//     std::cout << "Test value " << z.val << std::endl;
//     std::cout << "Gradient: " << z.grad.transpose() << std::endl;


//     // std::cout << "Angle: " << angle.val << std::endl;
//     // std::cout << "Gradient: " << g.transpose() << std::endl;

//     // std::cout << "Test value " << xy.transpose() << std::endl;
//     // std::cout << "Gradient: " << xy.grad.transpose() << std::endl;
//     return 0;
// }


// int main()
// {
//     // Differentiate w.r.t. 3 variables: x, y, z
//     using AD = TinyAD::Double<3>;

//     // Active variables at evaluation point (x=1, y=2, z=3)
//     Eigen::Vector3<AD> v = AD::make_active({1.0, 2.0, 3.0});
//     AD x = v[0], y = v[1], z = v[2];

//     // f(x,y,z) = x² + 2xy + sin(z)
//     AD f = x*x + 2.0*x*y + sin(z);

//     // Results
//     double            val  = f.val;    // f(1,2,3)
//     Eigen::Vector3d   grad = f.grad;   // [∂f/∂x, ∂f/∂y, ∂f/∂z]
//     Eigen::Matrix3d   H    = f.Hess;   // 3×3 Hessian matrix

//     std::cout << "f    = " << val  << "\n";   // 5.1411
//     std::cout << "grad = " << grad.transpose() << "\n"; // [6, 2, cos(3)]
//     std::cout << "H =\n"  << H    << "\n";
// }

// int main()
// {
//     // F : ℝ³ → ℝ²
//     // F(x,y,z) = [ x²  +  sin(y)       ]
//     //            [ x·y  -  exp(z)  +  z ]

//     constexpr int n_vars = 3;   // inputs
//     constexpr int n_out  = 2;   // outputs

//     using AD = TinyAD::Double<n_vars>;

//     // Evaluation point
//     Eigen::Vector3<AD> v = AD::make_active({1.0, 2.0, 0.5});
//     AD x = v[0], y = v[1], z = v[2];

//     // Define each output component
//     AD f0 = x*x + sin(y);
//     AD f1 = x*y - exp(z) + z;

//     // Values
//     Eigen::Vector2d F_val(f0.val, f1.val);

//     // Jacobian  J[i,j] = ∂fᵢ/∂xⱼ   (2 × 3 matrix)
//     Eigen::Matrix<double, n_out, n_vars> J;
//     J.row(0) = f0.grad.transpose();  // [∂f0/∂x, ∂f0/∂y, ∂f0/∂z]
//     J.row(1) = f1.grad.transpose();  // [∂f1/∂x, ∂f1/∂y, ∂f1/∂z]

//     // Individual Hessians per output  (each 3 × 3)
//     Eigen::Matrix3d H0 = f0.Hess;
//     Eigen::Matrix3d H1 = f1.Hess;

//     std::cout << "F =\n" << F_val << "\n\n";
//     std::cout << "J =\n" << J     << "\n\n";
//     std::cout << "H0=\n"<< H0    << "\n\n";
//     std::cout << "H1=\n"<< H1    << "\n";
// }

// #include <TinyAD/Scalar.hh>
// Template over scalar type -> same code runs as double or AD
// template<typename Scalar, int M, int N>
// Eigen::Matrix<Scalar, M, 1> my_func(const Eigen::Matrix<Scalar, N, 1>& v)
// {
//     Scalar x = v[0], y = v[1], z = v[2];
//     Eigen::Matrix<Scalar, M, 1> out;
//     out << x * x + sin(y),
//            x * y - exp(z) + z;
//     return out;
// }

// int main()
// {
//     using AD_3 = TinyAD::Double<3>;

//     Eigen::Matrix<AD_3, 3, 1> x_ad = AD_3::make_active({1.0, 2.0, 0.5});
//     Eigen::Matrix<AD_3, 2, 1> F = my_func<AD_3, 2, 3>(x_ad);

//     // Jacobian assembled from per-row gradients
//     Eigen::Matrix<double, 2, 3> J;
//     for (int i = 0; i < 2; ++i)
//         J.row(i) = F[i].grad.transpose();
//     std::cout << "J =\n" << J << "\n\n";
// }

// template<int M, int N>

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

int main() {
    NewtonSolver<4, 4> solver;
    Eigen::Vector<double, 4> guess;
    
    guess << 310.0, -8.0, 5. ,25.0;
    solver.set_initial_guess(guess);


    // solver.print_initial_guess();
    // std::cout << "Initial settings: " << solver.get_settings().max_iterations << ", " << solver.get_settings().tol << std::endl;
    
    // solver.Jacobian(solver.initial_guess, my_func_ad);
    // if (solver.J)
    //     std::cout << "Jacobian at initial guess:\n" << *solver.J << std::endl;

    // std::cout << "Evaluating function at initial guess:" << solver.initial_guess << std::endl;
    solver.solve(Wood);
    if (solver.success)    {
        std::cout << "Solver converged in " << solver.iterations << " iterations." << std::endl;
        std::cout << "Solution: " << solver.solution.transpose() << std::endl;
        std::cout << "Function evaluation at transpose " << solver.F0.transpose() << std::endl;

    } else {
        std::cout << "Solver did not converge after " << solver.iterations << " iterations." << std::endl;
    }
    // auto G = my_func_ad(guess);
    // solver.assign_F0(G);
    // std::cout << solver.F0 << std::endl;
    return 0;
}