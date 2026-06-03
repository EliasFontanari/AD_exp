#include <TinyAD/Scalar.hh>
#include <iostream>
#include <cmath>

struct NewtonSolverSettings
{
    int max_iterations = 1000;
    double tol = 1e-9;
    // Armijo line search parameters
    double gamma = 0.1;
    double beta = 0.8;
    double t_max = 1.0;
};

template<int N, int M>  // N = number of variables, M = number of equations
class NewtonSolver
{public:
    using AD_N = TinyAD::Double<N>;
    using FuncSig = Eigen::Matrix<AD_N, M, 1>(*)(const Eigen::Matrix<AD_N, N, 1>&);

    NewtonSolverSettings settings;
    bool success = false;
    int iterations = 0;
    Eigen::Vector<double, N> solution = Eigen::Matrix<double, N, 1>::Zero();;
    Eigen::Matrix<AD_N, N, 1> initial_guess =
        AD_N::make_active(Eigen::Matrix<double, N, 1>::Zero());
    Eigen::Matrix<double, M, N>* J = nullptr;
    Eigen::Matrix<double, M, 1> F0 = Eigen::Matrix<double, M, 1>::Zero();       
    
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

    void assign_F0(const Eigen::Matrix<AD_N, M, 1> & F0_active) 
    {
        for (int i = 0; i < M; ++i)
            this->F0[i] = F0_active[i].val;
    }

   void assign_M_AD_vector(const Eigen::Matrix<AD_N, M, 1> & vec_source, Eigen::Matrix<double, M, 1> & vec_dest) 
    {   
        for (int i = 0; i < M; ++i)
            vec_dest[i] = vec_source[i].val;
    }

    void assign_solution(const Eigen::Matrix<AD_N, N, 1> & sol) 
    {   
        for (int i = 0; i < N; ++i)
            this->solution[i] = sol[i].val;
    }

    bool multivariate_check_minor(Eigen::Matrix<double, M, 1> & lhs , Eigen::Matrix<double, M, 1> & rhs) 
    {
        for (int i = 0; i < M; ++i) {
            if (lhs[i] > rhs[i]) {
                return false;
            } 
        }
        return true; 
    }

    double merit_fun(const Eigen::Matrix<AD_N, N, 1> & x, FuncSig func) {
        Eigen::Matrix<double, M,1> tmp;
        assign_M_AD_vector(func(x), tmp);
        return 0.5*pow(tmp.norm(),2);
    }

    double armijo_search(Eigen::Matrix<AD_N, N, 1> x, Eigen::Vector<double, N> d, FuncSig func) {
        int iter = 0;
        double t = settings.t_max;
        bool accept_t_step = true;
        Eigen::Matrix<double, M,1> lhs, rhs, val_at_x;
        assign_M_AD_vector(func(x +t*d), lhs); // the sum of AD vector and standard vector gives AD vector;
        assign_M_AD_vector(func(x) + settings.gamma * t * ((*J)*d) , rhs);
        assign_M_AD_vector(func(x) , val_at_x);
        Eigen::Matrix<double, N,1> merit_grad = (*J).transpose() * val_at_x;
        // while(!multivariate_check_minor(lhs,rhs) && iter < 50) {
        while (!(merit_fun(x+t*d,func) - merit_fun(x,func) - settings.gamma*t*merit_grad.transpose()*d <= 0)) {
            t *= settings.beta;
            assign_M_AD_vector(func(x + t*d),lhs); // the sum of AD vector and standarf vector gives AD vector;
            assign_M_AD_vector(func(x) + settings.gamma * t * ((*J)*d) , rhs);

            std::cout << "Current t " << t << std::endl;
            iter += 1;
        }
        return t;
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
                
                auto F0_active = func(x);
                assign_F0(F0_active);
                if ((this->F0).norm() < settings.tol)
                {
                    success = true;
                    assign_solution(x);
                    std::cout << "Converged in " << iterations << " iterations." << std::endl;
                    return;
                }
                Jacobian(x, func);
                auto lu = J->fullPivLu();
                Eigen::Vector<double, N> d0 = -lu.solve(F0);
                // std::cout << "d at step " << iterations << " has value " << d0 << std::endl; 
                double t = armijo_search(x,d0,func); 
                x += t*d0;
                std::cout << "t Armijo line search at step " << iterations << " has value " << t << std::endl; 
                iterations++;
            }
        }
        else
        {
            success = true;
            assign_solution(initial_guess);
        }
        
    }
};