/**
 * \file Newton_solver.hh
 * \brief Newton solver with Armijo line search using TinyAD.
 */

#include <TinyAD/Scalar.hh>
#include <iostream>
#include <cmath>

/*! Simple struct to hold solver settings */
struct NewtonSolverSettings
{
    int max_iterations = 1000; /**< Maximum number of iterations */
    double tol = 1e-9;         /**< Tolerance for convergence */
    // Armijo line search parameters
    double gamma = 1e-4; /**< Gamma for Armijo line search */
    double beta = 0.5;   /**< Beta for Armijo line search, parameters that multiplies t to update it at each iteration during line search */
    double t_max = 1.0;  /**< Maximum step size and initial value for Armijo line search */
};

/**
 * @brief Newton solver for systems of nonlinear equations.
 *
 * It is a class that exploits automatic differentiation to compute the Jacobian matrix of
 * the system of equations at each iteration, and uses it to compute the descent direction, employing Armijo line search.
 * It is templatized on the number of variables and equations, and it require the user to provide the function that computes the system of equations to be differentiated.
 * @tparam N number of variables
 * @tparam M number of equations
 */
template <int N, int M>
class NewtonSolver
{
public:
    using AD_N = TinyAD::Double<N>; // add false to not keep track of Hessian                                                                     ///< AD scalar type with N-dimensional gradient. It is the type that take into account the variables respect to which we differentiate.
    using FuncSig = void (*)(const Eigen::Matrix<AD_N, N, 1> &, Eigen::Matrix<AD_N, N, 1> &);           ///< It is the signature of the function that computes the system of equations to be differentiated. It takes as input an Eigen vector of AD scalars, and outputs an Eigen vector of AD scalars, which is the evaluation of the system of equations at the input point. The user has to provide a function with this signature to compute the system of equations to be solved.
    using FuncSigDouble = void (*)(const Eigen::Matrix<double, N, 1> &, Eigen::Matrix<double, N, 1> &); ///< Signature for the same function asa bove, but without automatic differentiation. Used when just an evaluation of the function is needed.

protected:
    NewtonSolverSettings settings;
    bool success = false;
    int iterations = 0;
    Eigen::Vector<double, N> solution = Eigen::Matrix<double, N, 1>::Zero();
    Eigen::Matrix<double, N, 1> *initial_guess = new Eigen::Matrix<double, N, 1>(Eigen::Matrix<double, N, 1>::Zero());
    Eigen::Matrix<double, M, N> *J = nullptr;
    Eigen::Matrix<double, M, 1> *F_val = nullptr;
    Eigen::Matrix<AD_N, M, 1> *F_val_ad = nullptr;
    Eigen::Matrix<double, N, 1> d = Eigen::Matrix<double, N, 1>::Zero(); // descent direction

    // Armijo preallocated vectors
    Eigen::Matrix<double, M, 1> *lhs_armijo = nullptr;
    Eigen::Matrix<double, M, 1> *rhs_armijo = nullptr;

public:
    /**
     * @brief Returns whether the solver converged successfully.
     * @return true if the solver converged, false otherwise.
     */
    bool get_success() const
    {
        return success;
    }

    /**
     * @brief Returns the number of iterations performed by the solver.
     * @return The number of iterations performed.
     */
    int get_iterations() const
    {
        return iterations;
    }

    /**
     * @brief Returns the number of iterations performed by the solver.
     * @return The number of iterations performed.
     * @todo Also add possibility of setting Armijo line search parameters.
     */
    void set_settings(int max_iterations, double tol)
    {
        settings.max_iterations = max_iterations;
        settings.tol = tol;
    }

    /**
     * @brief Returns the settings struct.
     */
    NewtonSolverSettings get_settings() const
    {
        return settings;
    }

    /**
     * @brief Provide an initial guess to the solver.
     */
    void set_initial_guess(Eigen::Matrix<double, N, 1> &guess)
    {
        this->initial_guess = &guess; // no new allocation
    }

    void print_initial_guess() const
    {
        Eigen::Matrix<double, N, 1> vals;
        for (int i = 0; i < N; ++i)
            vals[i] = initial_guess[i].val;
        std::cout << "Initial guess values: " << vals.transpose() << std::endl;
    }

    /**
     * @brief Function to compute Jacobian matrix with automatic differentiation.
     *
     * It does not have return but it assigns the values of the Jacobian to the J attribute.
     * It also keep track of the value of the function at the current input point, which is stored in F_val_ad.
     *
     * @param input The point at which to compute the Jacobian, it must be an eigen vector of AD scalars.
     * @param func The function that computes the system of equations. It must have the signature FuncSig.
     */
    void Jacobian(
        const Eigen::Matrix<AD_N, N, 1> &input,
        FuncSig func)
    {
        if (this->J == nullptr) // if J attribute nullptr, allocate it.
        {
            this->J = new Eigen::Matrix<double, M, N>();
        }
        if (this->F_val_ad == nullptr)
        {
            this->F_val_ad = new Eigen::Matrix<AD_N, M, 1>();
        }

        func(input, *F_val_ad); // assign to F_val_ad the values, gradient and hessians at current input point.

        for (int i = 0; i < M; ++i)                            // compose the Jacobian.
            this->J->row(i) = (*F_val_ad)[i].grad.transpose(); // write directly
    }
    /**
     * @brief Returns the solution found by the solver.
     */
    const Eigen::Ref<const Eigen::MatrixXd> get_solution() const
    {
        return solution;
    }

    /**
     * @brief Solve the problem, given the function to be solved, coherent in size with the instantiated solver.
     */
    void solve(FuncSig func, FuncSigDouble func_double)
    {
        Eigen::Matrix<AD_N, N, 1> x = AD_N::make_active(*initial_guess);
        // func(x, *F_val_ad); // compute F_val_ad at the initial guess
        if (F_val == nullptr)
        {
            F_val = new Eigen::Matrix<double, M, 1>(); // allocate once, no copy
        }
        func_double(*initial_guess, *F_val); // compute F_val at the initial guess
        double norm_F = (*F_val).norm();
        Eigen::Matrix<double, N, 1> *x_double = nullptr;
        assign_AD_vector(x, x_double);

        if (norm_F > settings.tol)
        {
            while (iterations < settings.max_iterations && !success)
            {

                Jacobian(x, func);
                auto lu = J->fullPivLu();
                d = -lu.solve(*F_val);

                double t = armijo_search(*x_double, d, func_double, *F_val);
                // double t = 1.0; // for now, no line search
                x += t * d;
                assign_AD_vector(x, x_double); // update x_double for the Armijo line search in the next iteration

                std::cout << "t Armijo line search at step " << iterations << " has value " << t << std::endl;
                iterations++;
                // update F0 at the new x
                func(x, *F_val_ad);
                assign_F(*F_val_ad);
                norm_F = (*F_val).norm();
                if (norm_F < settings.tol)
                {
                    success = true;
                    assign_solution(x);
                    std::cout << "Converged in " << iterations << " iterations. Norm of F: " << norm_F << std::endl;
                    return;
                }
                std::cout << "Iteration " << iterations << ", norm of F: " << norm_F << std::endl;
            }
        }
        else
        {
            success = true;
            assign_solution(*initial_guess);
        }
    }

protected:
    /**
     * @brief Function to store the value of an eigen vector of AD scalars into the attribute F_val that stores the value of the system of equations.
     * @param F_active The eigen vector of AD scalars to be stored in F_val.
     */
    void assign_F(const Eigen::Matrix<AD_N, M, 1> &F_active)
    {
        if (F_val == nullptr)
        {
            F_val = new Eigen::Matrix<double, M, 1>(); // allocate once, no copy
        }
        for (int i = 0; i < M; ++i)
            (*F_val)[i] = F_active[i].val;
    }

    /**
     * @brief Function to store the value of an eigen vector of AD scalars into an eigen vector of doubles. It is templetized on the size of the vectors.
     * @param vec_source The eigen vector of AD scalars to be stored in vec_dest.
     * @param vec_dest The eigen vector of doubles where the values will be stored.
     */
    template <int Size>
    void assign_AD_vector(
        const Eigen::Matrix<AD_N, Size, 1> &vec_source,
        Eigen::Matrix<double, Size, 1> *&vec_dest)
    {
        if (vec_dest == nullptr)
            vec_dest = new Eigen::Matrix<double, Size, 1>();

        for (int i = 0; i < Size; ++i)
            (*vec_dest)[i] = vec_source[i].val;
    }

    /**
     * @brief Function to store the value of an eigen vector of AD scalars into the solution attribute, which is an eigen vector of doubles.
     */
    void assign_solution(const Eigen::Matrix<AD_N, N, 1> &sol)
    {
        for (int i = 0; i < N; ++i)
            this->solution[i] = sol[i].val;
    }

    /**
     * @brief Computes the merit function value: phi(x) = 0.5*norm_2(F(x))^2
     * @param x The point at which to evaluate the merit function.
     * @param tmp Temporary vector for intermediate computations.
     * @param func The function for which to compute the merit function.
     * @return The value of the merit function.
     */
    double merit_fun(const Eigen::Matrix<double, N, 1> &x, Eigen::Matrix<double, M, 1> *&tmp, FuncSigDouble func)
    {
        func(x, *tmp);
        return 0.5 * pow((*tmp).norm(), 2);
    }

    /**
     * @brief Performs Armijo line search to find a suitable step size t that satisfies the Armijo condition.
     * @param x The current point in the optimization.
     * @param d The descent direction found witht he Jacobian and the resolution of the LU system.
     * @param func The function for which to perform the line search.
     * @param f_val_x The value of the function at the current point x, used to compute the Armijo condition.
     * @return The step size t that satisfies the Armijo condition.
     */
    double armijo_search(Eigen::Matrix<double, N, 1> &x, Eigen::Vector<double, N> &d, FuncSigDouble func, Eigen::Vector<double, N> &f_val_x)
    {
        if (lhs_armijo == nullptr)
            lhs_armijo = new Eigen::Matrix<double, M, 1>();
        if (rhs_armijo == nullptr)
            rhs_armijo = new Eigen::Matrix<double, M, 1>();
        int iter = 0;
        double t = settings.t_max;
        Eigen::Matrix<double, N, 1> merit_grad = (*J).transpose() * (f_val_x);
        double merit_incr = settings.gamma * merit_grad.dot(d);
        double merit_val = merit_fun(x, rhs_armijo, func);
        while (!(merit_fun(x + t * d, lhs_armijo, func) - merit_val - t * merit_incr <= 0) && t > 1e-6)
        {
            t *= settings.beta;
            // assign_M_AD_vector(func(x + t*d),lhs_armijo); // the sum of AD vector and standarf vector gives AD vector;
            // assign_M_AD_vector(func(x) + settings.gamma * t * ((*J)*d) , rhs_armijo);
            std::cout << "Current t " << t << std::endl;
            iter += 1;
        }
        return t;
    }
};