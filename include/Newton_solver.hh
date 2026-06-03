#include <TinyAD/Scalar.hh>
#include <iostream>

struct NewtonSolverSettings
{
    int max_iterations = 1000;
    double tol = 1e-6;
};

template<int N, int M>  // N = number of variables, M = number of equations
class NewtonSolver
{public:
    using AD_N = TinyAD::Double<N>;
    using FuncSig = Eigen::Matrix<AD_N, M, 1>(*)(const Eigen::Matrix<AD_N, N, 1>&);

    NewtonSolverSettings settings;
    bool success = false;
    int iterations = 0;
    Eigen::Vector<double, N>* solution = nullptr;
        Eigen::Matrix<AD_N, N, 1> initial_guess =
            AD_N::make_active(Eigen::Matrix<double, N, 1>::Zero());
    Eigen::Matrix<double, M, N>* J = nullptr;
    Eigen::Matrix<AD_N, M, 1>* F0 = nullptr;
    
    void set_settings(int max_iterations, double tol)
    {
        settings.max_iterations = max_iterations;
        settings.tol = tol;
    }

    NewtonSolverSettings get_settings() const
    {
        return settings;
    }

    void set_initial_guess(const Eigen::Matrix<double, N, 1>& guess)
    {
        this->initial_guess = AD_N::make_active(guess);
    }

    void print_initial_guess() const
    {
        Eigen::Matrix<double, N, 1> vals;
        for (int i = 0; i < N; ++i)
            vals[i] = initial_guess[i].val;
        std::cout << "Initial guess values: " << vals.transpose() << std::endl;
    }

    void Jacobian(
        const Eigen::Matrix<AD_N, N, 1>& input,
        FuncSig func)
    {
        Eigen::Matrix<AD_N, M, 1> F = func(input);
        Eigen::Matrix<double, M, N> Jac;
        for (int i = 0; i < M; ++i)
            Jac.row(i) = F[i].grad.transpose();

        this->J = new Eigen::Matrix<double, M, N>(Jac);
    }

    void solve(FuncSig func)
    {
        if (func(initial_guess).norm() > settings.tol)
        {
            Eigen::Matrix<double, N, 1> x_double;
            for (int i = 0; i < N; ++i)
                x_double[i] = initial_guess[i].val;
            Eigen::Matrix<AD_N, N, 1> x = AD_N::make_active(x_double);
            while (iterations < settings.max_iterations && !success)
            {
                // Eigen::Matrix<double, N, 1> x_double::Zero();
                // for (int i = 0; i < N; ++i)
                //     x_double[i] = x[i].val;
                
                auto F0 = func(x);
                if (F0.norm() < settings.tol)
                {
                    // success = true;
                    // solution = new Eigen::Vector<double, N>(x);
                    std::cout << "Converged in " << iterations << " iterations." << std::endl;
                    return;
                }
                // Jacobian(x, func);
                // auto lu = J->fullPivLu();
                // Eigen::Vector<double, N> d0 = -lu.solve(F0);
                // x += d0;
                iterations++;
            }
        }
        else
        {
            success = true;
            // this->solution = new Eigen::Vector<double, N>(initial_guess);
        }
        
    }
};